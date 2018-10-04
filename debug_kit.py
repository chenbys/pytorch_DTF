import matplotlib.pyplot as plt


def show_img_tensor(img_t):
    img = img_t.detach().cpu().numpy()
    plt.imshow(img)
    plt.show()
