

###INFORME LABORATORIO 2 CAPTURA DE VOCES
####CAPTURA DE LAS VOCES 	
Se utilizaron tres dispositivos de captura de audio para grabar a tres personas hablando sobre distintos temas, simulando un ambiente festivo. Las grabaciones incluyeron tanto las voces individuales como el ruido ambiental presente en la sala de grabación. Tras obtener estas grabaciones, se procedió a la captura y procesamiento de las señales utilizando el siguiente código
####captura señal

	# %% inicializacion
	import librosa
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn.decomposition import FastICA
	import soundfile as sf

	# %% niveles de cuantificacion
	bits = 16
	cbits = 2 ** bits

	print('niveles de cuantificacion', cbits)

	# %% Carga los tres archivos de audio
	y1, sr1 = librosa.load('audio 1.mp3', sr=None)
	y2, sr2 = librosa.load('audio 2.mp3', sr=None)
	y3, sr3 = librosa.load('audio 3.mp3', sr=None)

	# Imprimir la frecuencia de muestreo de cada archivo
	print('Frecuencia de muestreo del audio 1:', sr1)
	print('Frecuencia de muestreo del audio 2:', sr2)
	print('Frecuencia de muestreo del audio 3:', sr3)

	# Crea un vector de tiempo para cada señal en segundos
	t1 = np.linspace(0, len(y1) / sr1, len(y1))
	t2 = np.linspace(0, len(y2) / sr2, len(y2))
	t3 = np.linspace(0, len(y3) / sr3, len(y3))

	# %% graficar señales
	# Señal de Audio 1
	plt.plot(t1, y1)
	plt.title('Señal de Audio 1')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud')
	plt.show()

	# Señal de Audio 2
	plt.plot(t2, y2)
	plt.title('Señal de Audio 2')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud')
	plt.show()

	# Señal de Audio 3
	plt.plot(t3, y3)
	plt.title('Señal de Audio 3')
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud')
	plt.show()
El código proporcionado muestra la captura de señales obtenidas de tres fuentes diferentes. Para cada una de estas señales, se realizó un análisis individual con el objetivo de determinar aspectos clave como los niveles de cuantificación y las frecuencias de muestreo de las grabaciones. Además, se generaron vectores de tiempo correspondientes a cada señal para facilitar su representación gráfica y para analizar la duración de cada grabación.
####CALCULO DE LOS SNR DE CADA SEÑAL RESPECTO AL RUIDO GRBADO EN LA SALA
####calculo SNR
	#SNR
	#cargar ruido de cada fuente
	r1, rr1 = librosa.load('ruido 1.mp3', sr=None)
	r2, rr2 = librosa.load('ruido 2.mp3', sr=None)
	r3, rr3 = librosa.load('ruido 3.mp3', sr=None)

	# Imprimir la frecuencia de muestreo de cada archivo
	print('Frecuencia de muestreo del ruido 1:', rr1)
	print('Frecuencia de muestreo del ruido 2:', rr2)
	print('Frecuencia de muestreo del ruido 3:', rr3)

	# Crea un vector de tiempo para cada señal en segundos
	tr1 = np.linspace(0, len(r1) / rr1, len(r1))
	tr2 = np.linspace(0, len(r2) / rr2, len(r2))
	tr3 = np.linspace(0, len(r3) / rr3, len(r3))

	#graficar ruidos

	# Señal de Audio 1

	plt.plot(tr1, r1)
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud')
	plt.show()
	# Señal de Audio 2

	plt.plot(tr2, r2)
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud')
	plt.show()
	# Señal de Audio 3

	plt.plot(tr3, r3)
	plt.xlabel('Tiempo (s)')
	plt.ylabel('Amplitud')
	plt.show()
	#%%calcular SNR
	#potencia ruidos 
	def pot(s):
		potencia = np.mean(s**2)
		return potencia

	potencia_sig1=pot(y1)
	potencia_sig2=pot(y2)
	potencia_sig3=pot(y3)


	#calcular potencias del ruido 

	potencia_rui1=pot(r1)
	potencia_rui2=pot(r2)
	potencia_rui3=pot(r3)

	#calcular SNR
	def snr(ps,pr):
		Snr=10*np.log10(ps/pr)
		return Snr

	SNR1=snr(potencia_sig1,potencia_rui1)
	SNR2=snr(potencia_sig2,potencia_rui2)
	SNR3=snr(potencia_sig3,potencia_rui3)

	print('SNR1: ',SNR1)
	print('SNR2: ',SNR2)
	print('SNR3: ',SNR3)
En el código proporcionado, se llevan a cabo varios cálculos. Primero, se cargan las grabaciones del ruido provenientes de las tres fuentes diferentes. Luego, se calcula la potencia tanto de las señales de audio como del ruido, con el objetivo de determinar la relación señal-ruido (SNR) para cada una de las grabaciones. Para facilitar el proceso, se crearon funciones específicas dentro del código.
####ANALISIS TEMPORAL DE LA SEÑAL
Para ello se realizo el cálculo de distintas características est5adisticas como la media desviación estándar y coeficiente de variación.
Código
#### Calcular medias y desviaciones estándar
	media1 = np.mean(y1)
	media2 = np.mean(y2)
	media3 = np.mean(y3)

	Dstd1 = np.std(y1)
	Dstd2 = np.std(y2)  
	Dstd3 = np.std(y3)  

	# Función para calcular el coeficiente de variación
	def cv(med, ds):
		CV = ds / med
		return CV

	# Calcular coeficientes de variación
	Cv1 = cv(media1, Dstd1)
	Cv2 = cv(media2, Dstd2)
	Cv3 = cv(media3, Dstd3)

	# Imprimir los valores
	print(f"Media 1: {media1}")
	print(f"Desviación estándar 1: {Dstd1}")
	print(f"Coeficiente de variación 1: {Cv1}")

	print(f"Media 2: {media2}")
	print(f"Desviación estándar 2: {Dstd2}")
	print(f"Coeficiente de variación 2: {Cv2}")

	print(f"Media 3: {media3}")
	print(f"Desviación estándar 3: {Dstd3}")
	print(f"Coeficiente de variación 3: {Cv3}")

Además del código se adjuntan los valores obtenidos de cada variable desde los SNR hasta los datos estadísticos 
Valores obtenidos
Frecuencia en Hz
Frecuencia de muestreo del audio 1: 44100
Frecuencia de muestreo del audio 2: 44100
Frecuencia de muestreo del audio 3: 44100
Frecuencia de muestreo del ruido 1: 44100
Frecuencia de muestreo del ruido 2: 44100
Frecuencia de muestreo del ruido 3: 44100
SNR en DB
SNR1:  34.44856643676758
SNR2:  37.1731162071228
SNR3:  27.67429828643799
Media 1: -1.5222786714730319e-05
Desviación estándar 1: 0.08010348677635193
Coeficiente de variación 1: 5262.07763671875
Media 2: 2.5590709356038133e-06
Desviación estándar 2: 0.07873646169900894
Coeficiente de variación 2: 30767.595703125
Media 3: 0.000302812026347965
Desviación estándar 3: 0.06597543507814407
Coeficiente de variación 3: 217.87586975097656
Imágenes graficas



####Graficos capturas de grabaciones en el dominio del tiempo
[![Figure-2024-09-05-120836-0.png](https://i.postimg.cc/WpBxJ0JZ/Figure-2024-09-05-120836-0.png)](https://postimg.cc/Wt75BF8p)

[![Figure-2024-09-05-120836-1.png](https://i.postimg.cc/VsnfHqnf/Figure-2024-09-05-120836-1.png)](https://postimg.cc/Th2vKWtF)

[![Figure-2024-09-05-120836-2.png](https://i.postimg.cc/hjwXbck2/Figure-2024-09-05-120836-2.png)](https://postimg.cc/62r6BJ5Z)

####ruidos
[![Figure-2024-09-05-120836-3.png](https://i.postimg.cc/63GjtLNz/Figure-2024-09-05-120836-3.png)](https://postimg.cc/tZptz6JV)
[![Figure-2024-09-05-120836-4.png](https://i.postimg.cc/d0ZpVK28/Figure-2024-09-05-120836-4.png)](https://postimg.cc/DWFBxHYZ)
[![Figure-2024-09-05-120836-5.png](https://i.postimg.cc/0jmxhwQJ/Figure-2024-09-05-120836-5.png)](https://postimg.cc/gLYfZr3Y)
Image:

####ANALISIS ESPECTRAL 
Se utilizo la transformada rápida de Fourier para el análisis espectral de las señales, utilizando el siguiente código
# Análisis frecuencial con FFT

	def plot_fft(signal, sr, title):
		n = len(signal)
		yf = np.fft.fft(signal)
		xf = np.fft.fftfreq(n, 1/sr)

		plt.figure(figsize=(12, 6))
		plt.plot(xf[:n//2], np.abs(yf[:n//2]))  # Solo la mitad positiva del espectro
		plt.title(title)
		plt.xlabel('Frecuencia [Hz]')
		plt.ylabel('Amplitud')
		plt.grid(True)
		plt.show()

	# Graficar FFT para cada señal de audio
	plot_fft(y1, sr1, 'FFT del Audio 1')
	plot_fft(y2, sr2, 'FFT del Audio 2')
	plot_fft(y3, sr3, 'FFT del Audio 3')

Se generaron gráficos de las señales en el dominio de la frecuencia. Para ello, el código utilizó funciones de la librería numpy para realizar la Transformada Rápida de Fourier (FFT). Además, se creó un vector de frecuencias utilizando la frecuencia de muestreo y el número de muestras de la señal, el cual se empleó para el eje x de los gráficos.
####Gráficos de Fourier
[![Figure-2024-09-05-120836-6.png](https://i.postimg.cc/cHC5KGGx/Figure-2024-09-05-120836-6.png)](https://postimg.cc/cgqcjjJ2)

[![Figure-2024-09-05-120836-7.png](https://i.postimg.cc/L6JDj3WT/Figure-2024-09-05-120836-7.png)](https://postimg.cc/rKL5kxM0)

[![Figure-2024-09-05-120836-8.png](https://i.postimg.cc/MHv4LM6J/Figure-2024-09-05-120836-8.png)](https://postimg.cc/ZvzVBqvw)

####METODO DE SEPARACION DE FUENTES 
Por ultimo se utiliza el método de separación de fuentes ICA para aislar las voces de interés de las tres personas de la grabación mediante el siguiente código.
####ICA
	min_length = min(len(y1), len(y2), len(y3))

	y1 = y1[:min_length]
	y2 = y2[:min_length]
	y3 = y3[:min_length]

	X = np.vstack([y1, y2, y3])


	# Aplicar ICA
	ica = FastICA(n_components=3)
	S = ica.fit_transform(X.T)  # Transponer X para que cada columna sea una señal
	S1, S2, S3 = S[:, 0], S[:, 1], S[:, 2]



	# Guardar las señales separadas en archivos WAV
	sf.write('separada_1.wav', S1, sr1)
	sf.write('separada_2.wav', S2, sr1)
	sf.write('separada_3.wav', S3, sr1)

	print('Archivos WAV guardados. Puedes reproducirlos con cualquier reproductor de audio.')
	#%%SNR señales separadas
	potencia_separadas1=pot(S1)
	potencia_separadas2=pot(S2)
	potencia_separadas3=pot(S3)
	potenciaS2=pot(S2)
	#respecto a la señal separada 2 S2

	SNR_separadas1=snr(potenciaS2,potencia_sig1)
	SNR_separadas2=snr(potenciaS2,potencia_sig2)
	SNR_separadas3=snr(potenciaS2,potencia_sig3)
	#Snr respecto a cada señal original
	print('snr de la señal separada S2: ',SNR_separadas1)
	print('snr de la señal separada S2: ',SNR_separadas2)
	print('snr de la señal separada S2: ',SNR_separadas3)

Primero, se indexaron las señales de las tres fuentes en una matriz para llevar a cabo el aislamiento correspondiente, utilizando las funciones presentadas en el código. Luego, cada voz aislada se almacenó en una variable y se guardó para su reproducción futura. Además, se calculó la relación señal-ruido (SNR) respecto a la segunda voz aislada (S2), comparándola con cada una de las fuentes originales de grabación. A continuación, se presentan los valores obtenidos para el SNR.

snr de la señal separada S2:  21.92697286605835Db
snr de la señal separada S2:  22.07648515701294Db
snr de la señal separada S2:  23.612265586853027Db

####librerias utlizadas
	import librosa
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn.decomposition import FastICA
	import soundfile as sf

