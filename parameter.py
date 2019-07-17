f = open("./data_japan/vocab.txt", "r",encoding="utf-8")
vocab = f.read().splitlines()

letters = vocab

num_classes = len(letters) + 1

img_w, img_h = 512, 64

# Network parameters
batch_size = 128
val_batch_size = 128

downsample_factor = 4
max_text_len = 125