#### train with textzoom
 #python3 main.py --batch_size=1024 --STN --mask --gradient --vis_dir='vis' --arch tsrn

## train with textzoomsyn x2,x4
#python3 main.py --batch_size=25 --syn --mask --gradient --vis_dir='test47-hranSeriesLaplace-lr0.0001-syn' \
 #   --demo_dir=='test47-hranSeriesLaplace-lr0.0001-syn' --arch hranSeriesLaplace

## train with textzoomReal x2
#python3 main.py --batch_size=20 --STN --mask --vis_dir='test18-hran-lr0.0001' \
#  --demo_dir=='test18-hran-lr0.0001' --arch hran

## train with icdar2015
#python3 main.py --batch_size=128 --icdar2015 --mask --gradient --vis_dir='test22-tsrn-lr0.0001-icdar2015' \
#--demo_dir=='test22-tsrn-lr0.0001-icdar2015' --arch tsrn


##### tensorboard
#tensorboard --logdir=runs


##### demo
#python3 main.py --demo --demo_dir='./images/1/' --resume='./ckpt/result/test3/tsrn2model_best.pth' --STN --mask --rec aster


##### test with textzoom
## syn x2,x4
python3 main.py --batch_size=20  --test  --test_data_dir='./dataset/textZoom/test/hard/' \
   --resume='./ckpt/test41-srres-lr0.0001-syn/srres2model_best.pth' --syn --mask --gradient  --vis_dir='test41-srres-lr0.0001-syn' --rec aster --arch srres

#Bicubic
# python3 main.py --batch_size=20 --test --test_data_dir='./dataset/textZoom/test/hard/' \
 #   --syn --vis_dir='test38-bicubic-x4' --rec aster --arch bicubic

## real x2
#python3 main.py --batch_size=20 --test --test_data_dir='./dataset/textZoom/test/medium/' \
#  --resume='./ckpt/test6-tsrn-lr0.0001/tsrn2model_best.pth' --STN --mask --gradient --vis_dir='test6-tsrn-lr0.0001' --rec aster --arch tsrn

##### test with icdar2015
#python3 main.py --batch_size=1 --test --test_data_dir='./dataset/ICDAR2015/DATA/TEST' \
# --resume='./ckpt/test15-hranSeries-lr0.0001/hranSeries2model_best.pth' --icdar2015 --STN --mask --gradient --vis_dir='test15-hranSeries-lr0.0001' --rec aster --arch hranSeries

## syn x2,x4
#python3 main.py --batch_size=1 --test --test_data_dir='./dataset/ICDAR2015/DATA/TEST' \
#   --resume='./ckpt/test31-hranSeriesLaplace-lr0.0001-laplaceatten/hranSeriesLaplace4model_best.pth' --icdar2015 --syn --mask  --gradient --vis_dir='test31-hranSeriesLaplace-lr0.0001-laplaceatten-ICDAR2015' --rec aster --arch hranSeriesLaplace

#Bicubic
# python3 main.py --batch_size=1 --test --test_data_dir='./dataset/ICDAR2015/DATA/TEST' \
#    --syn --vis_dir='test38-bicubic-ICDAR2015-x4' --icdar2015 --rec aster --arch bicubic