mkdir data
wget https://www.mllp.upv.es/europarl-st/v1.1.tar.gz -O ./data/europarl-st.tar.gz 
tar -xzvf ./data/europarl-st.tar.gz  -C ./data/
mv ./data/v1.1 ./data/europarl-st