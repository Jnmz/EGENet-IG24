import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models.networks import BASE_Transformer

def attention_map(map,title):
    map = map.detach().cpu().numpy()[0]

    # 对所有通道取平均得到综合的单通道特征图

    map = np.mean(map, axis=0)
    # 将综合的单通道特征图展示
    plt.imshow(map, cmap='jet')
    # plt.colorbar()

    plt.savefig('vis_feature_map/'+title)

# 定义图像的预处理转换
transformss = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建模型并加载预训练权重
model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                         with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
model.to('cuda')
checkpoint = torch.load("checkpoints/LEVIR_BASE_EDGE_SA4_HF3_compose/best_ckpt.pt", map_location="cuda")
model.load_state_dict(checkpoint['model_G_state_dict'])
model.cpu()

# 加载图像并进行预处理
A = Image.open("LEVIR-CD-256/A/train_100_14.png")
B = Image.open("LEVIR-CD-256/B/train_100_14.png")
img_a = transformss(A).unsqueeze(0)
img_b = transformss(B).unsqueeze(0)

# 获取 self.map1，即 HF 后的特征图
model.eval()
with torch.no_grad():
    _ = model(img_a, img_b)  # 这里需要运行一次 forward，让模型计算 self.map1

attention_map(model.resnet_b1_map1,"resnet_b1_map1")
attention_map(model.resnet_b2_map1,"resnet_b2_map1")
attention_map(model.resnet_b3_map1,"resnet_b3_map1")
attention_map(model.resnet_b4_map1,"resnet_b4_map1")
attention_map(model.resnet_b1_hf_map1,"resnet_b1_hf_map1")
attention_map(model.resnet_b2_hf_map1,"resnet_b2_hf_map1")
attention_map(model.resnet_b3_hf_map1,"resnet_b3_hf_map1")
attention_map(model.resnet_b4_hf_map1,"resnet_b4_hf_map1")
attention_map(model.resnet_b1_map2,"resnet_b1_map2")
attention_map(model.resnet_b2_map2,"resnet_b2_map2")
attention_map(model.resnet_b3_map2,"resnet_b3_map2")
attention_map(model.resnet_b4_map2,"resnet_b4_map2")
attention_map(model.resnet_b1_hf_map2,"resnet_b1_hf_map2")
attention_map(model.resnet_b2_hf_map2,"resnet_b2_hf_map2")
attention_map(model.resnet_b3_hf_map2,"resnet_b3_hf_map2")
attention_map(model.resnet_b4_hf_map2,"resnet_b4_hf_map2")
attention_map(model.resnet_map1,"resnet_map1")
attention_map(model.resnet_map2,"resnet_map2")

attention_map(model.edge_encoder_map1,"edge_encoder_map1")
attention_map(model.edge_encoder_map2,"edge_encoder_map2")
attention_map(model.edge_encoder_hf_map1,"edge_encoder_hf_map1")
attention_map(model.edge_encoder_hf_map2,"edge_encoder_hf_map2")
attention_map(model.edge_decoder_map1,"edge_decoder_map1")
attention_map(model.edge_decoder_map2,"edge_decoder_map2")
attention_map(model.edge_decoder_hf_map1,"edge_decoder_hf_map1")
attention_map(model.edge_decoder_hf_map2,"edge_decoder_hf_map2")
# attention_map(model.transformer_encoder_map1,"transformer_encoder_map1")
# attention_map(model.transformer_encoder_map2,"transformer_encoder_map2")
attention_map(model.transformer_decoder_map1,"transformer_decoder_map1")
attention_map(model.transformer_decoder_map2,"transformer_decoder_map2")
attention_map(model.edge_transformer_decoder_map1,"edge_transformer_decoder_map1")
attention_map(model.edge_transformer_decoder_map2,"edge_transformer_decoder_map2")
attention_map(model.edge_transformer_decoder_hf_map1,"edge_transformer_decoder_hf_map1")
attention_map(model.edge_transformer_decoder_hf_map2,"edge_transformer_decoder_hf_map2")
attention_map(model.final_map1,"final_map1")
attention_map(model.final_map2,"final_map2")
attention_map(model.diff,"absolute result")
attention_map(model.edge_map,"edge_map")
attention_map(model.change_map,"change_map")