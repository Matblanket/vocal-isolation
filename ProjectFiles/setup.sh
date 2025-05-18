sudo apt update
sudo apt install aria2 -y
aria2c -s 16 -x 16 https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip -d /
unzip -q ../wham_noise.zip -d /wham/