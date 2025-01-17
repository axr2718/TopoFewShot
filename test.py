from models.swinv2 import SwinV2
import timm

swinaruv2raru = SwinV2(num_classes=5,
                       pretrained=False,
                       drop_rate=0.0,
                       drop_rate_path=0.0)


swinv2 = timm.create_model('swinv2_small_window16_256.ms_in1k')
for name, module in swinv2.named_parameters():
    print(name)