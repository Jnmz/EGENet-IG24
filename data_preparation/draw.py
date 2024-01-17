import numpy as np
import matplotlib.pyplot as plt


def draw(file, label):
    len = file.size
    plt.plot(np.linspace(start=0, stop=len - 1, num=len, ), file, label=label)


train_base = np.load('LEVIR_BASE/train_acc.npy',
                     encoding="latin1", allow_pickle=True)
train_edge = np.load('LEVIR_BASE_EDGE/train_acc.npy',
                     encoding="latin1", allow_pickle=True)
train_sa1 = np.load('LEVIR_BASE_EDGE_SA1_HF1_simple/train_acc.npy',
                    encoding="latin1", allow_pickle=True)
train_sa4 = np.load('LEVIR_BASE_EDGE_SA4_HF1_simple/train_acc.npy',
                    encoding="latin1", allow_pickle=True)
train_hf3 = np.load('LEVIR_BASE_EDGE_SA4_HF3_simple/train_acc.npy',
                    encoding="latin1", allow_pickle=True)
train_hf3_ = np.load('LEVIR_BASE_EDGE_SA4_HF3_compose/train_acc.npy',
                     encoding="latin1", allow_pickle=True)

val_base = np.load('LEVIR_BASE/val_acc.npy',
                   encoding="latin1", allow_pickle=True)
val_edge = np.load('LEVIR_BASE_EDGE/val_acc.npy',
                   encoding="latin1", allow_pickle=True)
val_sa1 = np.load('LEVIR_BASE_EDGE_SA1_HF1_simple/val_acc.npy',
                  encoding="latin1", allow_pickle=True)
val_sa4 = np.load('LEVIR_BASE_EDGE_SA4_HF1_simple/val_acc.npy',
                  encoding="latin1", allow_pickle=True)
val_hf3 = np.load('LEVIR_BASE_EDGE_SA4_HF3_simple/val_acc.npy',
                  encoding="latin1", allow_pickle=True)
val_hf3_ = np.load('LEVIR_BASE_EDGE_SA4_HF3_compose/val_acc.npy',
                   encoding="latin1", allow_pickle=True)

draw(train_base, 'BASE')
draw(train_edge, 'BASE_EDGE')
draw(train_sa1, 'BASE_EDGE_SA1_HF1')
draw(train_sa4, 'BASE_EDGE_SA4_HF1')
draw(train_hf3, 'BASE_EDGE_SA4_HF3')
draw(train_hf3_, 'BASE_EDGE_SA4_HF3_compose')

plt.title('train_acc')
plt.legend()
plt.savefig("train_acc.png")
plt.show()
plt.clf()
draw(val_base, 'BASE')
draw(val_edge, 'BASE_EDGE')
draw(val_sa1, 'BASE_EDGE_SA1_HF1')
draw(val_sa4, 'BASE_EDGE_SA4_HF1')
draw(val_hf3, 'BASE_EDGE_SA4_HF3')
draw(val_hf3_, 'BASE_EDGE_SA4_HF3_compose')
plt.title('val_acc')
plt.legend()
plt.savefig("val_acc.png")
plt.show()
