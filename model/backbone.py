import open_clip
import torch
import timm
def Openclip(name,pretrained):
    model,_,_ = open_clip.create_model_and_transforms(model_name=name,pretrained=pretrained)
    inp=torch.rand(1,3,224,224)
    _,embed_size=model(inp)[0].shape
    return model,embed_size

def backbone_timm(name):
    model = timm.create_model(name,
    pretrained=True,
    num_classes=0,
    )
    inp=torch.rand(1,3,224,224)
    out=model(inp)
    embed_size=model(inp)[0].shape[0]
    return model,embed_size
if __name__=="__main__":
    # model_name='ViT-B-16'
    # pretrained='laion2b_s34b_b88k'
    # model,size=Openclip(model_name,pretrained)
    model,size=backbone_timm('convnextv2_tiny.fcmae_ft_in22k_in1k')

    print(size)