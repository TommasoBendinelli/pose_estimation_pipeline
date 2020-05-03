from YOLACT.yolact import Yolact

print('Loading model...', end='')
net = Yolact()
net.load_weights(args.trained_model)
net.eval()