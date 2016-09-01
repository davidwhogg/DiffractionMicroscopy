import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy
import glob

def create_video_from_model_images():
	"""
	function loads the images of the model and concatenates them to a video file
	"""
	images_path = glob.glob("./model_*.png")
	print("number of image files to process: %s" % len(images_path))

	fig = plt.figure()
	ims = []
	for i in xrange(len(images_path)):
		print i
		img=mpimg.imread(images_path[i])
		lum_img = img[:,:,0]
		ims.append((plt.pcolormesh(lum_img[60:500, 180:640], cmap='gray'),))
	print("loaded all images")

	im_ani = animation.ArtistAnimation(fig, ims, interval=20, repeat_delay=3000, blit=True)
	print("created the video")
	im_ani.save('./model_vid.mp4')
	print("done, the video is ready!")