import cv2, os
import numpy as np
import matplotlib.pyplot as plt

face_database = "orl-faces"

# Select HAAR cascades for detecting face
face_cascade1 = cv2.CascadeClassifier(
	'haarcascade_frontalface_alt_tree.xml')
face_cascade2 = cv2.CascadeClassifier(
	'haarcascade_frontalface_alt2.xml')

# Select HAAR cascades for detecting two eyes
lefteye_cascade = cv2.CascadeClassifier(
	'haarcascade_mcs_lefteye.xml')
righteye_cascade = cv2.CascadeClassifier(
	'haarcascade_mcs_righteye.xml')

# Select HAAR cascades for detecting mouth
mouth_cascade1= cv2.CascadeClassifier(
	'haarcascade_mcs_mouth.xml')
mouth_cascade2= cv2.CascadeClassifier(
	'haarcascade_smile.xml')

# This function is used to get distance of two points and return it
def get_dst(p1, p2):
	p1 = np.array(p1)
	p2 = np.array(p2)
	return np.sqrt(np.sum(np.square(p2 - p1)))

# This function is used to get midpoint of two points and return the
# coordinates in the format of numpy array
def get_med(p1, p2):
	p1 = np.array(p1)
	p2 = np.array(p2)
	return (p1 + p2) / 2.0

# This function is used to draw the scattergram.
def draw_scatt(datalist):
	label = [int(List[0].split("_")[0].split("s")[1])
			 for List in datalist]
	ratio = [List[1] for List in datalist]
	plt.figure(figsize=(18.5, 10.5))
	plt.plot(label, ratio, 'b*') # show the points with *
	plt.xlabel('Label of People')
	plt.ylabel('Value of Ratio')
	plt.xticks(np.arange(42)) # set the step length of abscissa
	plt.ylim(0.0, 2) # set the range of ordinates
	plt.title("Distribution of People's Biometric")
	plt.legend()
	plt.show()

# This function is used to get the features in detecting face, eyes,
# and mouth. It returns a mark of whether the features are captured,
# which is followed by three other returned value indicate the position
# of left eye, right eye, and mouth.
def detector(img):

	# First, initialize the default value of some local variables
	get_lefteye = 0 # local variable of mark of detection of left eye
	get_righteye = 0 # local variable of mark of detection of right eye
	get_mouth = 0 # local variable of mark of detection of mouth
	flag = 'False' # local variable of mark of detection both of above
	pt1 = 0 # point of left eye pupil
	pt2 = 0 # point of right eye pupil
	pt3 = 0 # point of central of mouth

	# Configure the cascade of facial detector. 3 parameters used are:
	# 1. The img indicates the frame to be detected.
	# 2. The second parameter is the rate of reducing the scale of
	# detection patch.
	# 3. The third parameter is the time of reducing the scale to
	# certify a feature.
	faces = face_cascade1.detectMultiScale(img, 1.1, 3)
	if faces == ():

		# If not detected, change the optional detector
		faces = face_cascade2.detectMultiScale(img, 1.1, 3)
		if faces == ():

			# If still undetectable, return the default value
			return flag, pt1, pt2, pt3

	for (x, y, w, h) in faces:

		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

		roi_color = img[y:y + h, x:x + w]

		# Get the set of coordinates from the left eye, right eye
		# and mouth detectors.
		leye = lefteye_cascade.detectMultiScale(roi_color)
		reye = righteye_cascade.detectMultiScale(roi_color)
		mouth = mouth_cascade1.detectMultiScale(roi_color)

		for (lex, ley, lew, leh) in leye:	

			# For every set detected, filter it with restriction
			# of position:
			# Make presumption that the left eye is in the area
			# of the 3/5 right-top region of the photo.
			if (lex > int(0.4 * w)) & ((ley + leh) < int(0.6 * h)):

				get_lefteye = 1 # set the mark

				# Display the left eye with a rectangle, and set
				# the midpoint to the pt1.
				# If detected, stop. If not, move to the next set
				cv2.rectangle(roi_color, (lex, ley),
							  (lex + lew, ley + leh), (0, 255, 0), 1)
				pt1 = (lex + int(0.5 * lew), ley + int(0.5 * leh))
				break

		for (rex, rey, rew, reh) in reye:

			# Same as described above. Filter with restriction.
			if ((rex + rew) < int(0.6 * w)) & ((rey + reh) < int(0.6 * h)):

				get_righteye = 1
				cv2.rectangle(roi_color, (rex, rey),
							  (rex + rew, rey + reh), (0, 0, 255), 1)
				pt2 = (int(rex + 0.5 * rew), int(rey + 0.5 * reh))
				break

		if mouth == ():

			# If not detected, change the optional detector
			mouth = mouth_cascade2.detectMultiScale(roi_color)
			if mouth == ():

				# If still undetectable, return the default value
				return	flag, pt1, pt2, pt3
		
		for (mx, my, mw, mh) in mouth:

			# Same as described above. Make presumption that
			# the mouth is in the area of the 2/5 bottom
			# region of the photo.
			if my > int(0.6 * h):

				get_mouth = 1

				cv2.rectangle(roi_color, (mx, my),
							  (mx + mw, my + mh), (255, 0, 0), 1)
				pt3 = (int(mx + 0.5 * mw), int(my + 0.5 * mh))
				break

		# If all the features are detected, return the mark valued by 1.
		if (get_lefteye == 1) & (get_righteye == 1) & (get_mouth == 1):
			flag = 'True'
		return flag, pt1, pt2, pt3

##################################Main body of the project##############

# First initialize variables of mark and list of data sets of detection
detected = 'False'	
detected_list = []

# Start a traversal of every image from every folder of the people
for p in os.listdir(face_database):

	# Enter into the folders whose name starts with 's'
	if p.startswith('s'):

		# Take the names of photos and joint the full path of it, then
		# save it in a list named im_paths
		im_paths = [os.path.join(face_database, p, im)
					for im in os.listdir(os.path.join(face_database, p))
					if not im.startswith('._')]

		# When get the list, start another traversal of processing the photos
		for im_path in im_paths:
			image = cv2.imread(im_path)
			detected, lpt, rpt, mpt = detector(image) # get 4 returned values

			# To set the data sets, a label of people and a ratio are needed.
			# The labels are defined using the name of folder, and joint it
			# with a '_', then adds the name of photo without suffix.
			label = p + '_' + str(int(os.path.split(im_path)[1].split(".")[0]))
			if detected == 'True':

				# If the features of photo are detected. Figure out the ratio.
				print 'Detected: ', label, [detected, lpt, rpt, mpt]
				dst1 = get_dst(lpt, rpt) # distance of two eyes
				med = get_med(lpt, rpt) # figure out the midpoint of two eyes
				dst2 = get_dst(med, mpt) # distance of mouth and the midpoint
				ratio = dst1 / dst2

				# Append the list of data sets with new set of label and ratio
				detected_list.append([label, ratio])
			else:
				print 'Not Detected: ', label

print '400 finished,', len(detected_list), 'captured.' # the number of detected

draw_scatt(detected_list) # show the scattergram
