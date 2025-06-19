import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize

st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("./model.keras")
    return model


def preprocessdata(file_path,target_shape=(148,148)):
    data = []
    audio_data,sample_rate = librosa.load(file_path,sr = None)
    chunk_duration = 4 
    overlap_duration = 2 
    chunk_samples = chunk_duration*sample_rate
    overlap_samples = overlap_duration*sample_rate
    num_chunks = int(np.ceil((len(audio_data)-chunk_samples)/(chunk_samples-overlap_samples))) + 1
    for i in range(num_chunks):
        start = i*(chunk_samples-overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        spectrogram = librosa.feature.melspectrogram(y=chunk,sr=sample_rate)
        spectrogram = resize(np.expand_dims(spectrogram,axis=-1),target_shape)
        data.append(spectrogram)

    return np.array(data)



def ModelPrediction(x_test):
    model = load_model()
    y_predicted = model.predict(x_test)
    predicted_categories = np.argmax(y_predicted,axis=1)
    unique_elements,counts = np.unique(predicted_categories,return_counts=True)
    max_count = np.max(counts)
    max_element = unique_elements[counts == max_count]
    return max_element[0]



st.sidebar.title("DashBoard")
app_node = st.sidebar.selectbox("Select Page", ["Home" , "Project Description" , "Prediction Of Music"])

if(app_node == "Home"):
    st.markdown(''' ## Welcome to the.\n
    ## Music Genre Classification System!  ''')
    image_path = "94476Music Genre Classification Project.png"
    st.image(image_path , use_container_width=True)
    st.markdown("""
    **Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

    ### How It Works
    1. **Upload Audio:** Go to the **Prediction of Music** page and upload an audio file.
    2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
    3. **Results:** View the predicted genre along with related information.

    ### Why Choose Us?
    - **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
    - **User-Friendly:** Simple and intuitive interface for a smooth user experience.
    - **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

    ### Get Started
    Click on the **Prediction Of Music** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

    ### About Us
    Learn more about the project, our team, and our mission on the **Project Description** page.
    """)



#About Project
elif(app_node=="Project Description"):
    st.markdown("""
                ### About Project
                Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

                This data hopefully can give the opportunity to do just that.

                ### About Dataset
                #### Content
                1. **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
                2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
                3. **images original** - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. **2 CSV files** - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.
                """)
    

elif(app_node=="Prediction Of Music"):
    st.header("Model prediction")
    test_mp3 = st.file_uploader("Upload an audio file" , type={"mp3"})
    
    if test_mp3 is not None:
        filepathmp3 = "test music/"+test_mp3.name

    if(st.button("Play Audio")):
        st.audio(test_mp3)

    if(st.button("Predict Genre")):
        with st.spinner("...Please Wait"):
            x_test = preprocessdata(filepathmp3)
            result = ModelPrediction(x_test)
            st.balloons()
            label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
            st.markdown("**Model Prediction: The Genre Of This Audio Is :red[{}]**".format(label[result]))


