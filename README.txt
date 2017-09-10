# High-Performance-Computing
# Classification task on AT&T face database from Olivetti Research Laboratories
# The objective of this experiment is to demonstrate how the biometric
# (the ratio of the distance between the eyes of a person to the
# distance from eyes to the mouth) works in facial recognition.
#
# As to set a standard for evaluating this biometric, a large amount of
# pictures should be processed and analyzed. With regards to the fact
# that detection can be effect by lots of factors, I elaborated on this
# objective by transferring it into two dependent aspects:
# 1. Whether the ratios from the same person are intensive.
# 2. If yes, whether the ratio from any person is exclusive from those
# of others.
#
# I visualized these two aspects with a scattergram, which takes the
# labels of different people as abscissa while ratios as ordinates. The
# former aspect can be figured out by comparing lengthwise and the
# latter transversely.
#
# As we will see, the result would show that, when lengthwise compared,
# the ratios from the same person are relative intensive for a certain
# range, though there would be some particular points disjoining the
# concentration due to accuracy problem of detection. However, when
# compared transversely, ratios from some of people would appear
# exclusive and representative from those of others, because we can
# see that ratios from different people are in the same level in overall.
#
# Regarding to this, THE RATIO COULD NOT PROVIDE A GOOD BIOMETRIC.
#
# The data sets come from the results of processing the pictures in AT&T
# face database, which contains 10 gray head portraits with different
# features for every person out of 40. The database folder MUST be
# in the CURRENT DIRECTORY WITH THIS FILE.
#
# One of the permanent URL for the database is:
# ttp://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
#
# As a tool implemented in OpenCV, Haar cascades were used as detectors.
# The cascades used in this experiment are listed as follows:
# haarcascade_frontalface_alt_tree.xml, haarcascade_frontalface_alt2.xml,
# haarcascade_mcs_lefteye.xml, haarcascade_mcs_mouth.xml,
# haarcascade_mcs_righteye.xml, haarcascade_smile.xml
# These files MUST be in the CURRENT DIRECTORY WITH THIS FILE.
#
# ps:
# In some situation, detectors failed to capture features in the photos,
# though there was large range of choice of different detectors. I tried
# every one and chose one that performed relatively well and used some
# of others for option. If still undetectable, the picture would be put
# away from the valid data sets. So the number of data sets is much
# smaller than that of the face database. It is a matter of learning
# from the training sets and choosing algorithm.

