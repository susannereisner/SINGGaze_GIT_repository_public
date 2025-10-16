from scipy.stats import entropy
#relative entropy = KLdiv
from scipy.special import rel_entr

from filtering import *

def compute_spectrogram(x,rate): ##
    NFFT=len(x)
    b = x 
    f, B = signal.periodogram(b, rate,nfft =NFFT,scaling = 'density')
    return(f,B)


def resize_proportional(arr, n): ###resize array to be of length n
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)

def normalize(sig, rms_level=20):
    """
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - infile    (str) : input filename/path.
        - rms_level (int) : rms level in dB.
    """
    r = 10**(rms_level / 10.0)
    a = np.sqrt( (len(sig) * r**2) / np.sum(sig**2) )

    # normalize
    y = sig * a

    return y


def Wav2Envelope(wav,fcut):
	#get broadband amplitude envelope form the absolute value of the Hilbert tranform
	envelope = np.abs(hilbert(wav))
	#lowpass filter the envelope to f_cut using butterworth 
	envelope = butter_lowpass_filter(envelope,fcut,44100,5)
	return envelope


def get_Spectral_Flux(filename,ws,hs):
	x_total, rate = librosa.load(filename ,sr = 44100) 
	#normalise each audio file
	x_total = normalize(x_total,20)
	N = len(x_total)
	rate = 44100
	#compute number of windows based on lenght, ws and hs.
	nb_window = math.floor((N - (ws*rate))/(hs*rate)) + 1
	print('nb window = ', nb_window)
	SF = np.zeros(nb_window)
	#set maximum frequency of interest to 3000Hz
	freq_max_of_interest = math.floor(3000 * ws)
	for window in range(nb_window):
		start_sample = math.floor(window * hs * rate)
		end_sample = start_sample + math.floor(ws*rate)
		
		#select first window
		x_sub = x_total[start_sample:end_sample]
		#compute spectrogram of that time window
		f,B_sub_previous = compute_spectrogram(x_sub,rate)
        
		start_sample = start_sample + math.floor(hs*rate)
		end_sample = end_sample + math.floor(hs*rate)
		#select second window
		x_sub = x_total[start_sample:end_sample]
		#compute spectrogram of that time window
		f,B_sub_next = compute_spectrogram(x_sub,rate)

		### plot consecutive spectrogram in same plot
        # plt.plot(f[0:freq_max_of_interest],B_sub_previous[0:freq_max_of_interest],label = 'fft previous')
        # plt.plot(f[0:freq_max_of_interest],B_sub_next[0:freq_max_of_interest],label = 'fft next')
        # plt.legend()
        # plt.show()
        
        # compute the spectral flux as the sum of the absolute differences in amplitude across frequency
		SF[window] = sum(abs(B_sub_previous[1:freq_max_of_interest] - B_sub_next[1:freq_max_of_interest]))

	return SF

'''
IMPORTANT! The tempo function is not very accurate. Instead, we've switched to using the MIR toolbox in Matlab.
'''
# def get_tempi(x,sr):
# 	#prior assumed distribution of the tempo = around 120bpm with 40bpm of std
# 	prior = scipy.stats.norm(loc=120, scale=40)
# 	onset_env = librosa.onset.onset_strength(y=x, sr=sr)
# 	tempi_1= librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
# 									hop_length = 512,start_bpm = 120,std_bpm = 40, max_tempo = 150,ac_size = 20,
#                                     aggregate=None,prior = prior)
# 	print("len(tempi_1)=",len(tempi_1))
# 	#low pass filtering the tempo at 50Hz using a 5th order butterworth filter
# 	tempi_1_filt = butter_lowpass_filter(tempi_1,50,44100,5)
# 	return tempi_1_filt


def get_f0_Praat(filename): ### get F0 using parselmouth (python implementation of Praat)
	X = parselmouth.Sound(filename)
	snd = parselmouth.Sound.extract_part(X)
	pitch = snd.to_pitch_ac(time_step = 0.01, pitch_floor= 70, very_accurate  = True, silence_threshold = 0.03, voicing_threshold = 0.5, octave_cost= 0.01, octave_jump_cost = 0.7, voiced_unvoiced_cost = 0.05, pitch_ceiling =800.0)
	pitch_values = pitch.selected_array['frequency']
	pitch_values[pitch_values==0] = np.nan
	return pitch_values

ws = 0.03   #ws = window size (in seconds)
hs = 0.02   #hs = hop size (in seconds)

# input directory containing the audio files
audio_dir = # insert path here 

# list of the participants to analyse
ppt_id_list = ['25','32','34','35','39','42','45','46','49','50','51','52',
               '53','54','56','57','58','60','61','62','63','64','65','66',
               '68','70','71','72','73','76','77','78','80','81','82','83',
               '84','85','86','87','88','89','90','91','92','93','94','97',
               '98','99','100','101','102','103','104','105','106','107','108',
               '109','110','111','112','113','114','115','116','117','118',
               '119','123','124','125','126']


for ppt_id in ppt_id_list:
    file_name = 'SING_' + ppt_id + '.wav'
    
    print(ppt_id)
    # extract SF (Spectral Flux)
    SF = get_Spectral_Flux(audio_dir+file_name,ws,hs)
    print("load audio")
    sig, rate = librosa.load(audio_dir+file_name ,sr = 44100)
    
    #extract the tempo
    # print('compute librosa tempi')
    # tempi = get_tempi(sig,rate)
    #extract the F0
    pitch = get_f0_Praat(audio_dir+file_name)
    
    print("normalise")
    #normalise the signal across participants
    sig = normalize(sig,20)
    print("wav to env")
    #extract the amplitude envelope and lowpassfilter it at 200Hz
    env = Wav2Envelope(sig,200)
    print("get duration")
    # Resize every time series so that they are at 100Hz (number of sample = duration*100)
    duration = librosa.get_duration(filename = audio_dir+file_name)
    SF =resize_proportional(SF,int(duration*100))
    print("resize")
