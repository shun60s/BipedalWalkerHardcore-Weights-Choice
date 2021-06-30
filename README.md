# BipedalWalkerHardcore-v2 Weights Choice 

## 概要  

強化学習の[RL A3C Pytorch Continuous](https://github.com/dgriff777/a3c_continuous/)を使って、
OpenAI Gym のBipedalWalkerHardcore-v2の障害の状態をLidarの情報からDNNを使って予測させ、２種類の重みを切り替える方法を学習させるもの、検討中のもの。  


## 使い方  

Lidarの情報から障害の状態を予測するDNNの学習。  

```
python3 main.py --workers 24 --env BipedalWalkerHardcoreStateout-v2
```
  
  
動作テスト。　1回分（-num-episodes 1）のみ。   
```
python3 gym_eval.py --env BipedalWalkerHardcoreStateout-v2  --num-episodes 1 --discrete-number 4
```



## 動作環境  

現在のBipedalWalkerのバージョンは３であるが、古いバージョン２を使っている。  
CPUのみ。  

- python 3.8.5 on Ubuntu 20.04.2 LTS
- torch==1.5.0+cpu
- torchvision==0.6.0+cpu
- numpy==1.21.0
- gym==0.15.3
- Box2d-py==2.3.8
- pyglet==1.3.2
- pyyaml==3.12
- setproctitle==1.1.10
- typing==3.7.4.3


## ライセンス  
Apache License 2.0  
オリジナルのa3c_continuousのライセンス文 LICENSE_ac3_continous.MD を参照のこと。   
カスタム環境についてはOpenAI gymのライセンス文 custom_env/LICENSE-OpenAI_gym.md を参照のこと。   

