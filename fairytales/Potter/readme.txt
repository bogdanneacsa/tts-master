###########################################################################
#                                                                         #
#                         Copyright (c) 2007                              #
#                        All Rights Reserved.                             #
#                                                                         #
#  Permission is hereby granted, free of charge, to use and distribute    #
#  this data set and its documentation without restriction, including     #
#  without limitation the rights to use, copy, modify, merge, publish,    #
#  distribute, sublicense, and/or sell copies of this work, and to        #
#  permit persons to whom this work is furnished to do so, subject to     #
#  the following conditions:                                              #
#   1. The data must retain the above copyright notice, this list of      #
#      conditions and the following disclaimer.                           #
#   2. Any modifications must be clearly marked as such.                  #
#   3. Original authors' names are not deleted.                           #
#   4. The authors' names are not used to endorse or promote products     #
#      derived from this data set without specific prior written          #
#      permission.                                                        #
#                                                                         #
#  THE CONTRIBUTORS TO THIS WORK DISCLAIM ALL WARRANTIES                  #
#  WITH REGARD TO THIS DATA SET, INCLUDING ALL IMPLIED WARRANTIES OF      #
#  MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE                     #
#  CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL      #
#  DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA     #
#  OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER      #
#  TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR       #
#  PERFORMANCE OF THIS DATA SET.                                          #
#                                                                         #
###########################################################################
#                                                                         #
# Author: Cecilia Alm (ebbaalm@uiuc.edu)                                  #
#                                                                         #
###########################################################################

0. Disclaimer

See the above copyright notice.

Note that you are downloading this corpus at your own risk. No guarantee is provided, e.g. regarding the goodness of the corpus nor towards any subsequent effects. You may use it free of charge, and modify it as you wish, but clearly specify modifications if you pass modified material on.

I. Introduction and contact information
In  addition to the stories' affect annotations in the subdirectory emmood, a few directories with related files are included (some degree of noise should be expected).

The affect data website is currently available at: http://www.linguistics.uiuc.edu/grads/ebbaalm/affectdata/ When this webpage moves in the future, a link will redirect to the new website. Please see that website for information on the corpus. Also, for efficiency reasons, updates, announcements, corrections, and additional information is likely be available there.

Also, my dissertation ("Affect in text and speech") and this document provide additional information on this corpus.

Texts by this author can be downloaded from Project Gutenberg.

My contact email: ebbaalm@uiuc.edu
Release: June, 2007 (v. 1)

II. Contents
The document tales.txt lists the basenames of the story files. The below directories each have the same number of files, with a sentence per line. The directories agreeID and agree-sent only contain a subset of the corpus. Text has undergone some preprocessing, and files with infix/suffix 'okpuncs' some additional sentence preprocessing, compared to the emmood file (see the affectdata website). Sentence IDs start on zero for each story.

DIRECTORY:	CONTENT OF EACH FILE IN THE DIRECTORY:

1. emmood  	Lists sentences with unmerged affect labels for two annotators (A and B). The label set for both Primary emotion (1em) and Mood were: Angry (A), Disgusted (D), Fearful (F), Happy (H), Neutral (N), Sad (Sa for 1em, abbrev. to S for Mood), Pos.Surprised (Su+ for 1em, abbrev. to + for Mood), and Neg.Surprised (Su- for 1em, abbrev. to - for Mood)
File suffix: .emmood
Format: SentID:SentID	1emLabelA:1emLabelB	MoodLabelA:MoodLabelB	Sent
Example: 0:0     N:N     N:N     Once upon a time there was a village shop.
			
2. sent		Lists sentences (some additional processing, as for all files with the okpuncs suffix/infix; also see the affectdata website). 
File suffix: .sent.okpuncs
Example: Once upon a time there was a village shop.

3. pos  	Lists sentences with part-of-speech tags (some additional processing, as for all files with the okpuncs suffix/infix; also see the affectdata website). 
File suffix: .sent.okpuncs.props.pos
Format: (Tag word):(Tag word) [...]
Example: (RB Once):(IN upon):(DT a):(NN time):(EX there):(AUX was):(DT a):(NN village):(NN shop):(. .)

4. agreeID  	Lists only sentence IDs with AFFECTIVE HIGH AGGREMENTS, i.e. sentences with four identical affects. The merged labelset was used: Angry-Disgusted (merged), Fearful, Happy, Sad, and Surprised (merged). Note that the HighAgree subcorpus concerned sentences with affective labels, i.e. sentences with four Neutral labels are NOT included!
File suffix: .agreeID
Format: SentID
Example: 35

5. agree-sent  	Lists only sentences with AFFECTIVE HIGH AGGREMENTS (see description for corresponding agreeID directory). The Affective Label Codes are: 2=Angry-Disgusted, 3=Fearful, 4=Happy, 6=Sad, 7=Surprised
File suffix: .agree
Format: SentID@AffectiveLabelCode@Sentence 
Example: 35@3@"It is very unpleasant, I am afraid of the police," said Pickles.
 

