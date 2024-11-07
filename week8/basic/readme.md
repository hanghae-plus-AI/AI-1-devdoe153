
### Train/Loss
[https://wandb.ai/devdoe/Hanghae99/reports/train-loss-24-11-07-14-18-38---VmlldzoxMDA2MzE1OQ](https://api.wandb.ai/links/devdoe/ket16l7v)

### rank 8, 128, 256 합산 train/Loss
![train_loss](https://github.com/user-attachments/assets/d422bd1c-fd8d-43db-a533-9ddee8a4631d)



![r_32](https://github.com/user-attachments/assets/d1118d94-a7cd-4df7-94ec-4e322137fef5)

```
# wandb 로그
# rank 8 Runtime
215.6822
# rank 8 학습전 메모리 점유율
1.2 GB
# rank 8 학습후 메모리 점유율
5.1 GB
```


![r_128](https://github.com/user-attachments/assets/704ee6b1-7d70-4e1d-bb82-ec1f5ba3abe1)
```
# wandb 로그
# rank 128 Runtime
221.2925
# rank 128 학습전 메모리 점유율
5.1 GB
# rank 128 학습후 메모리 점유율
5.4 GB
```


![r_256](https://github.com/user-attachments/assets/d5ff8501-837e-4a4a-b197-dbf21540e2e8)

```
# wandb 로그
# rank 256 Runtime
230.1958
# rank 256 학습전 메모리 점유율
5.4 GB
# rank 256 학습후 메모리 점유율
5.8 GB
```

### LoRA의 장단점 분석
- 위 결과를 보고 추론 한다면, rank 값(8, 128, 256)이 변경되어도 train/loss 변동 거의 동일하게 진행되는 것 같다.
- 최초에 파인튜닝을 했을 때 자원이 다소 확보 되야 하지만, 그 이후 파인튜닝을 진행했을 경우 runtime, 메모리 점유율이
  변동 되지 않고 사용 되는 것 같다.

