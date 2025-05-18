
#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on Libri2/3Mix datasets.
The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-libri2mix.yaml
> python train.py hparams/sepformer-libri3mix.yaml


The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both libri2mix and
libri3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.core import AMPConfig
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Brain class for speech enhancement training
class Seperation(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""
    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        if targets!=[]:
            targets = targets[
                :, randstart : randstart + self.hparams.training_signal_len, :
            ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""
        # Unpack lists and put tensors in the right device
        try:
            mix, _ = mix
        except:
            pass
        mix = mix.to(self.device)

        # Convert targets to tensor
        if len(targets) != 0:
            targets = torch.cat(
                [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
                dim=-1,
            ).to(self.device)
            with torch.no_grad():
                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        # print(est_source.shape,targets.shape)
        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the si-snr loss"""
        return self.hparams.loss(targets, predictions)


    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        # targets = [batch.vocals, batch.drums, batch.bass, batch.other]
        targets = [batch.vocals, batch.drums]
        predictions, targets = self.compute_forward(
            mixture, targets, sb.Stage.TRAIN
        )
        loss = self.compute_objectives(predictions, targets)
        loss = loss.mean()
        if (
            loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
        ):  # the fix for computational problems
            loss.backward()
            if self.hparams.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(),
                    self.hparams.clip_grad_norm,
                )
            self.optimizer.step()
        else:
            self.nonfinite_count += 1
            logger.info(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0.0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()


    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        # snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.vocals, batch.drums]
        # targets = [batch.vocals, batch.drums, batch.bass, batch.other]
        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(batch.name, mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(batch.name, mixture, targets, predictions)

        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]},
                min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w", newline="", encoding="utf-8") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.name
                    # targets = [batch.vocals, batch.drums, batch.bass, batch.other]
                    targets = [batch.vocals, batch.drums]
                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    cut_mix,_ = self.cut_signals(mixture,[])
                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [cut_mix] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    print(mixture_signal.shape)
                    print(targets.shape)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline
                    try:
                        # Compute SDR
                        sdr, _, _, _ = bss_eval_sources(
                            targets[0].t().cpu().numpy(),
                            predictions[0].t().detach().cpu().numpy(),
                        )
    
                        sdr_baseline, _, _, _ = bss_eval_sources(
                            targets[0].t().cpu().numpy(),
                            mixture_signal[0].t().detach().cpu().numpy(),
                        )

                        sdr_i = sdr.mean() - sdr_baseline.mean()

                        # Saving on a csv file
                        row = {
                            "snt_id": snt_id,
                            "sdr": sdr.mean(),
                            "sdr_i": sdr_i,
                            "si-snr": -sisnr.item(),
                            "si-snr_i": -sisnr_i.item(),
                        }
                        writer.writerow(row)

                        # Metric Accumulation
                        all_sdrs.append(sdr.mean())
                        all_sdrs_i.append(sdr_i.mean())
                        all_sisnrs.append(-sisnr.item())
                        all_sisnrs_i.append(-sisnr_i.item())
                    except ValueError as e:
                        # Catch potential mir_eval errors that might still occur in edge cases
                        print(f"Error processing sample {snt_id}: {e}")

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))
        
        

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create output folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):
            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            if len(targets)!=0:
                signal = targets[0, :, ns]
                signal = signal / signal.abs().max()
                save_file = os.path.join(
                    save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
                )
                torchaudio.save(
                    save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
                )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )
        
    def infer_audio(self, path):
        audio, info_rate = torchaudio.load(path)
        print(audio.shape)
        mix_sig = torchaudio.functional.resample(audio, info_rate, self.hparams.sample_rate)
        mix_sig=mix_sig[0,:]
        if mix_sig.ndim == 1:
             mix_sig = mix_sig.unsqueeze(0)
        print(mix_sig.shape)

        final_output = torch.empty(0, 2, device=self.device)


        with torch.no_grad():
            audio_len = mix_sig.shape[1]
            segment_len = self.hparams.training_signal_len

            for i in range(0, audio_len, segment_len):
                end_index = min(i + segment_len, audio_len)

                current_segment = mix_sig[:, i:end_index]

                current_segment = current_segment.to(self.device)
                print(current_segment.shape)
                est, targets = self.compute_forward(current_segment, [], sb.Stage.VALID)

                if est.ndim != 2 or est.shape[1] != 2:
                     if est.ndim == 3 and est.shape[2] == 2:
                         est = est.squeeze(0)
                     elif est.ndim == 3 and est.shape[1] == 2:
                          est = est.squeeze(0).transpose(0, 1)
                     else:
                        raise ValueError(f"Expected estimated segment shape to be (time, 2) or (batch, time, 2) or (batch, 2, time), but got {est.shape}")


                est = est.to(final_output.device)

                final_output = torch.cat((final_output, est), dim=0)
        
        self.save_audio("test_output", torch.unsqueeze(mix_sig,0).cpu(),[],torch.unsqueeze(final_output,0).cpu())






def dataio_prep(hparams):
    """Creates data processing pipeline"""
    import musdb
    mus_train = musdb.DB(root="/notebooks/musdb18",subsets="train", split='train')
    mus_valid = musdb.DB(root="/notebooks/musdb18",subsets="train", split='valid')
    mus_test = musdb.DB(root="/notebooks/musdb18", subsets="test")
    train_data = {}
    valid_data= {}
    test_data = {}
    i = 0;
    for track in mus_train:
        i+=1
        dataobj={}
        dataobj['track'] = track
        train_data[track.name] = dataobj

    i = 0;
    for track in mus_valid:
        i+=1
        dataobj={}
        dataobj['track'] = track
        valid_data[track.name] = dataobj

    i = 0;
    for track in mus_test:
        i+=1
        dataobj={}
        dataobj['track'] = track
        test_data[track.name] = dataobj

    datasets = [
        sb.dataio.dataset.DynamicItemDataset(train_data),
        sb.dataio.dataset.DynamicItemDataset(valid_data),
        sb.dataio.dataset.DynamicItemDataset(test_data)
    ]

    @sb.utils.data_pipeline.takes("track")
    @sb.utils.data_pipeline.provides("name","mix_sig", "vocals","drums","bass","other")
    def audio_pipeline_mix(track):
        name = track.name

        mix_sig = torch.from_numpy(track.audio.T).float()
        mix_sig= torchaudio.functional.resample(mix_sig,track.rate,hparams['sample_rate'])[1,:]

        vocals = torch.from_numpy(track.sources['vocals'].audio.T).float()
        vocals = torchaudio.functional.resample(vocals,track.rate,hparams['sample_rate'])[1,:]

        drums = torch.from_numpy(track.sources['drums'].audio.T).float()
        drums = torchaudio.functional.resample(drums,track.rate,hparams['sample_rate'])[1,:]

        bass = torch.from_numpy(track.sources['bass'].audio.T).float()
        bass = torchaudio.functional.resample(bass,track.rate,hparams['sample_rate'])[1,:]

        other = torch.from_numpy(track.sources['other'].audio.T).float()
        other= torchaudio.functional.resample(other,track.rate,hparams['sample_rate'])[1,:]


        return name,mix_sig, vocals,drums,bass,other

    # @sb.utils.data_pipeline.takes("s1_wav")
    # @sb.utils.data_pipeline.provides("s1_sig")
    # def audio_pipeline_s1(s1_wav):
    #     s1_sig = sb.dataio.dataio.read_audio(s1_wav)
    #     return s1_sig

    # @sb.utils.data_pipeline.takes("s2_wav")
    # @sb.utils.data_pipeline.provides("s2_sig")
    # def audio_pipeline_s2(s2_wav):
    #     s2_sig = sb.dataio.dataio.read_audio(s2_wav)
    #     return s2_sig

    # if hparams["num_spks"] == 3:

    #     @sb.utils.data_pipeline.takes("s3_wav")
    #     @sb.utils.data_pipeline.provides("s3_sig")
    #     def audio_pipeline_s3(s3_wav):
    #         s3_sig = sb.dataio.dataio.read_audio(s3_wav)
    #         return s3_sig

    # if hparams["use_wham_noise"]:

    #     @sb.utils.data_pipeline.takes("noise_wav")
    #     @sb.utils.data_pipeline.provides("noise_sig")
    #     def audio_pipeline_noise(noise_wav):
    #         noise_sig = sb.dataio.dataio.read_audio(noise_wav)
    #         return noise_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    # sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1unzip ../wham_noise.zip -d /wham/)
    # sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    # if hparams["num_spks"] == 3:
    #     sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)

    # if hparams["use_wham_noise"]:
    #     print("Using the WHAM! noise in the data pipeline")
    #     sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noise)

    # if (hparams["num_spks"] == 2) and hparams["use_wham_noise"]:
    #     sb.dataio.dataset.set_output_keys(
    #         datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
    #     )
    # elif (hparams["num_spks"] == 3) and hparams["use_wham_noise"]:
    #     sb.dataio.dataset.set_output_keys(
    #         datasets,
    #         ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"],
    #     )
    # elif (hparams["num_spks"] == 2) and not hparams["use_wham_noise"]:
    #     sb.dataio.dataset.set_output_keys(
    #         datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
    #     )
    # else:
    sb.dataio.dataset.set_output_keys(
        # datasets, ["name", "mix_sig", "vocals", "drums", "bass", "other"]
        datasets, ["name", "mix_sig", "vocals", "drums"]
    )

    return datasets[0], datasets[1], datasets[2]


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)


    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    # Data preparation
    train_data, valid_data, test_data = dataio_prep(hparams)

    # Brain class initialization
    separator = Seperation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # separator.infer_audio("/notebooks/musdb18/test/Al James - Schoolboy Facination.stem.mp4")


    # Training
    separator.fit(
        separator.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Eval
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)
