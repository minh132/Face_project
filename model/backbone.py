
import torch
import timm
# from AdaFace import build_model

# def Openclip(name,pretrained):
#     model,_,_ = open_clip.create_model_and_transforms(model_name=name,pretrained=pretrained)
#     inp=torch.rand(1,3,224,224)
#     _,embed_size=model(inp)[0].shape
#     return model,embed_size

def backbone_timm(name):
    model = timm.create_model(name,
    pretrained=True,
    num_classes=0,
    )
    inp=torch.rand(1,3,224,224)
    out=model(inp)
    embed_size=model(inp)[0].shape[0]
    return model,embed_size
# def backbone_ada(name):
#     backbone=build_model(model_name=name)
#     inp=torch.rand(2,3,112,112)
#     _,out=backbone(inp)[0].shape
#     return backbone,out
if __name__=="__main__":
    # model_name='ViT-B-16'
    # pretrained='laion2b_s34b_b88k'
    # model,size=Openclip(model_name,pretrained)
    model,size=backbone_ada('ir_101')
    print(size)