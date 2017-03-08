from PIL import Image
from resizeimage import resizeimage

fd_img = open('tongue.jpg', 'r')
img = Image.open(fd_img)
img = resizeimage.resize_thumbnail(img, [200, 200])
img.save('test-image-thumbnail.jpeg', img.format)
fd_img.close()
