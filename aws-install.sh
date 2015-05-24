#!/bin/bash
cd /root
sed -i '/Defaults*[requiretty,visiblepw]/ s/^/#/' /etc/sudoers

sudo apt-get update
sudo apt-get install openjdk-7-jre-headless s3cmd haveged pinentry-curses htop mercurial \
binutils gcc g++ autoconf make freeglut3-dev build-essential libx11-dev libxmu-dev \
libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev emacs24

# get nvidia opencl driver for Tesla M2050
# http://www.nvidia.com/Download/
mkdir tmp
wget -O tmp/nvidia-driver.run http://us.download.nvidia.com/XFree86/Linux-x86_64/310.32/NVIDIA-Linux-x86_64-310.32.run
chmod 755 tmp/NVIDIA-*
tmp/NVIDIA-Linux-x86_64-310.32.run -s

# get AMD SDK
# http://developer.amd.com/amd-license-agreement/?f=AMD-APP-SDK-v2.7-lnx64.tar

# Note, by installing Mesa, you may see linking errors against libGL.  This can be solved below:
# cd /usr/lib/
# sudo rm libGL.so
# sudo ln -s libGL.so.1 libGL.so

rmmod nvidia
modprobe nvidia

sudo nvidia-smi

wget -O tmp/lein https://raw.github.com/technomancy/leiningen/stable/bin/lein
mv tmp/lein /usr/bin/lein

(:import [java.nio ByteBuffer]
    [com.jogamp.opencl CLPlatform CLContext CLDevice CLDevice$Type])

(cl/init-state
    :context (CLContext/create (into-array [(-> (CLPlatform/listCLPlatforms) first (.getMaxFlopsDevice))]))
    :program (clu/resource-stream cl-program))


cd
git clone --depth 1 git://source.ffmpeg.org/ffmpeg
cd ffmpeg
./configure --enable-gpl --enable-libass --enable-libfaac --enable-libfdk-aac --enable-libmp3lame \
  --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libspeex --enable-librtmp --enable-libtheora \
  --enable-libvorbis --enable-libvpx --enable-libx264 --enable-nonfree --enable-version3
make
sudo checkinstall --pkgname=ffmpeg --pkgversion="7:$(date +%Y%m%d%H%M)-git" --backup=no \
  --deldoc=yes --fstrans=no --default
hash -r


--------------------------------------------------------------------------------------

sudo apt-get update
sudo apt-get -y install openjdk-7-jre-headless s3cmd haveged pinentry-curses htop git mercurial binutils gcc g++ autoconf automake freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev emacs24 tmux nginx

sudo apt-get remove ffmpeg x264 libav-tools libvpx-dev libx264-dev yasm
# enable multiverse in sources
sudo emacs /etc/apt/sources.list
sudo apt-get -y install autoconf automake build-essential checkinstall git libass-dev libfaac-dev   libgpac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev librtmp-dev libspeex-dev   libtheora-dev libtool libvorbis-dev pkg-config texi2html zlib1g-dev
cd
wget http://www.tortall.net/projects/yasm/releases/yasm-1.2.0.tar.gz
tar xzvf yasm-1.2.0.tar.gz
cd yasm-1.2.0
./configure
make
sudo checkinstall --pkgname=yasm --pkgversion="1.2.0" --backup=no   --deldoc=yes --fstrans=no --default
cd
git clone --depth 1 git://git.videolan.org/x264.git
cd x264
./configure --enable-static
make
sudo checkinstall --pkgname=x264 --pkgversion="3:$(./version.sh | \
  awk -F'[" ]' '/POINT/{print $4"+git"$5}')" --backup=no --deldoc=yes   --fstrans=no --default
cd
git clone --depth 1 git://github.com/mstorsjo/fdk-aac.git
cd fdk-aac
autoreconf -fiv
./configure --disable-shared
make
sudo checkinstall --pkgname=fdk-aac --pkgversion="$(date +%Y%m%d%H%M)-git" --backup=no   --deldoc=yes --fstrans=no --default
cd
git clone --depth 1 http://git.chromium.org/webm/libvpx.git
cd libvpx
./configure --disable-examples --disable-unit-tests
make
sudo checkinstall --pkgname=libvpx --pkgversion="1:$(date +%Y%m%d%H%M)-git" --backup=no   --deldoc=yes --fstrans=no --default
cd
git clone --depth 1 git://source.ffmpeg.org/ffmpeg
cd ffmpeg
./configure --enable-gpl --enable-libass --enable-libfaac --enable-libfdk-aac --enable-libmp3lame   --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libspeex --enable-librtmp --enable-libtheora   --enable-libvorbis --enable-libvpx --enable-libx264 --enable-nonfree --enable-version3
make
sudo checkinstall --pkgname=ffmpeg --pkgversion="7:$(date +%Y%m%d%H%M)-git" --backup=no   --deldoc=yes --fstrans=no --default
hash -r
rm -rf fdk-aac/ ffmpeg/ libvpx/ x264/ yasm-1.2.0/ yasm-1.2.0.tar.gz

mkdir ~/bin
wget -O ~/bin/lein https://raw.github.com/technomancy/leiningen/stable/bin/lein
chmod +x ~/bin/lein
lein
mkdir projects
s3cmd --configure
ssh-keygen
gpg --gen-key
gpg --default-recipient-self -e ~/.lein/credentials.clj > ~/.lein/credentials.clj.gpg



sudo service nginx start
sudo update-rc.d nginx defaults
sudo emacs /etc/init.d/nginx
sudo service nginx status
sudo service nginx
sudo service nginx configtest
nano /etc/ssh/sshd_config
sudo emacs /etc/ssh/sshd_config
reload ssh
sudo reload ssh
sudo emacs /etc/nginx/nginx.conf
ll
whereis ffmpeg

ll
mkdir tmp
mv AMD-APP-SDK-v2.7-lnx64.tar tmp/
cd tmp/
tar -xvf AMD-APP-SDK-v2.7-lnx64.tar
ll
sudo ./Install-AMD-APP.sh
man reboot
sudo reboot
