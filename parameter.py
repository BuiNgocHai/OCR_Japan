f = open("./data_japan/vocab.txt", "r",encoding="utf-8")
vocab = f.read().splitlines()

letters = vocab

num_classes = len(letters) + 1

img_w, img_h = 800, 64

# Network parameters
batch_size = 2
val_batch_size = 4

downsample_factor = 4
max_text_len = 125