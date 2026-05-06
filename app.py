# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import json

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="🌿 Plant Disease Detection",
#     page_icon="🌿",
#     layout="wide"
# )

# # ---------------- CUSTOM CSS ----------------
# st.markdown("""
# <style>
# .main-title {
#     font-size:50px;
#     font-weight:900;
#     text-align:center;
#     background: linear-gradient(90deg, #1b5e20, #2e7d32, #66bb6a);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     letter-spacing:4px;
#     margin-bottom:10px;
# }

# .sub-title {
#     font-size:28px;
#     text-align:center;
#     color:#2e7d32;
#     margin-bottom:40px;
# }

# .disease-name {
#     font-size:34px;
#     font-weight:bold;
#     text-align:center;
#     color:#1b5e20;
# }

# .result-box {
#     padding:25px;
#     border-radius:12px;
#     background-color:#e8f5e9;
#     text-align:center;
# }

# .info-box {
#     padding:25px;
#     border-radius:12px;
#     background-color:#f1f8e9;
#     font-size:18px;
#     line-height:1.8;
# }
# </style>
# """, unsafe_allow_html=True)



# st.markdown('<div class="main-title">🌿 AI PLANT DISEASE DETECTION</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-title">Smart Leaf Analysis Using Deep Learning</div>', unsafe_allow_html=True)
# st.write("")

# # ---------------- LANGUAGE SELECTOR ----------------
# language = st.selectbox(
#     "🌍 Select Language / भाषा निवडा",
#     ["English", "हिंदी", "मराठी"]
# )

# # ---------------- LOAD MODEL ----------------
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("plant_disease_model.h5")

# model = load_model()

# # ---------------- LOAD CLASS NAMES ----------------
# with open("class_names.json", "r") as f:
#     class_names = json.load(f)



# # ---------------- TRANSLATIONS ----------------
# translations = {
#     "English": {
#         "upload": "📷 Upload Leaf Image",
#         "choose_source": "Choose Source:",
#         "upload_image": "Upload Image",
#         "use_camera": "Use Camera",
#         "low_conf": "⚠ Low confidence. Please upload a clearer image.",
#         "info_title": "📖 Disease Information & Cure",
#         "about": "🦠 About the Disease",
#         "cure": "💊 Treatment & Prevention",
#         "complete": "✅ Analysis Complete"
#     },
#     "हिंदी": {
#         "upload": "📷 पत्ते की तस्वीर अपलोड करें",
#         "choose_source": "स्रोत चुनें:",
#         "upload_image": "तस्वीर अपलोड करें",
#         "use_camera": "कैमरा उपयोग करें",
#         "low_conf": "⚠ कम विश्वसनीयता। कृपया साफ तस्वीर अपलोड करें।",
#         "info_title": "📖 रोग की जानकारी और उपचार",
#         "about": "🦠 रोग के बारे में",
#         "cure": "💊 उपचार और बचाव",
#         "complete": "✅ विश्लेषण पूर्ण हुआ"
#     },
#     "मराठी": {
#         "upload": "📷 पानाचा फोटो अपलोड करा",
#         "choose_source": "पर्याय निवडा:",
#         "upload_image": "फोटो अपलोड करा",
#         "use_camera": "कॅमेरा वापरा",
#         "low_conf": "⚠ कमी खात्री. कृपया स्पष्ट फोटो अपलोड करा.",
#         "info_title": "📖 रोगाची माहिती आणि उपाय",
#         "about": "🦠 रोगाबद्दल माहिती",
#         "cure": "💊 उपचार आणि प्रतिबंध",
#         "complete": "✅ विश्लेषण पूर्ण झाले"
#     }
# }



# # ---------------- MULTI-LANGUAGE DISEASE INFO ----------------
# disease_info = {

# # 🍎 Apple Apple Scab
# "Apple_Apple_scab": {

#     "English": {
#         "description": """
#         Apple scab is a fungal disease that causes dark, scabby lesions on leaves and fruits.
#         It spreads quickly in cool and wet weather conditions.
#         Severe infection can reduce fruit quality and cause early leaf drop.
#         """,
#         "cure": """
#         ✅ Apply fungicides like captan or sulfur.
#         ✅ Remove fallen infected leaves.
#         ✅ Improve air circulation through pruning.
#         ✅ Avoid overhead watering.
#         """
#     },

#     "हिंदी": {
#         "description": """
#         एप्पल स्कैब एक फंगल रोग है जो पत्तियों और फलों पर काले धब्बे बनाता है।
#         यह ठंडे और नम मौसम में तेजी से फैलता है।
#         गंभीर संक्रमण से फल की गुणवत्ता कम हो सकती है।
#         """,
#         "cure": """
#         ✅ कैप्टान या सल्फर का छिड़काव करें।
#         ✅ गिरी हुई संक्रमित पत्तियां हटाएं।
#         ✅ छंटाई करके हवा का प्रवाह बढ़ाएं।
#         ✅ ऊपर से पानी देने से बचें।
#         """
#     },

#     "मराठी": {
#         "description": """
#         ॲपल स्कॅब हा बुरशीजन्य रोग आहे जो पानांवर आणि फळांवर काळे डाग निर्माण करतो.
#         थंड आणि दमट हवामानात हा रोग जलद पसरतो.
#         जास्त संसर्ग झाल्यास फळांची गुणवत्ता कमी होते.
#         """,
#         "cure": """
#         ✅ कॅप्टान किंवा सल्फरची फवारणी करा.
#         ✅ गळलेली संक्रमित पाने काढा.
#         ✅ छाटणी करून हवा खेळती ठेवा.
#         ✅ वरून पाणी देणे टाळा.
#         """
#     }
# },

# # 🍎 Apple Black Rot
# "Apple_Black_rot": {

#     "English": {
#         "description": """
#         Black rot causes brown circular spots on leaves and rotting of fruits.
#         It spreads in warm and humid weather.
#         If untreated, it can severely damage the tree.
#         """,
#         "cure": """
#         ✅ Prune infected branches.
#         ✅ Remove infected fruits.
#         ✅ Apply copper-based fungicide.
#         ✅ Maintain tree health.
#         """
#     },

#     "हिंदी": {
#         "description": """
#         ब्लैक रॉट पत्तियों पर भूरे धब्बे और फलों में सड़न पैदा करता है।
#         यह गर्म और नम मौसम में फैलता है।
#         इलाज न करने पर पेड़ को गंभीर नुकसान हो सकता है।
#         """,
#         "cure": """
#         ✅ संक्रमित शाखाएं काटें।
#         ✅ संक्रमित फल हटा दें।
#         ✅ कॉपर फंगीसाइड का प्रयोग करें।
#         ✅ पौधे की उचित देखभाल करें।
#         """
#     },

#     "मराठी": {
#         "description": """
#         ब्लॅक रॉट पानांवर तपकिरी डाग आणि फळांमध्ये कुज निर्माण करतो.
#         उष्ण आणि दमट हवामानात हा रोग वाढतो.
#         उपचार न केल्यास झाडाचे मोठे नुकसान होते.
#         """,
#         "cure": """
#         ✅ संक्रमित फांद्या कापून टाका.
#         ✅ खराब फळे काढा.
#         ✅ कॉपर फंगीसाइड वापरा.
#         ✅ झाडाची योग्य निगा ठेवा.
#         """
#     }
# },

# # 🍎 Apple Cedar Apple Rust
# "Apple_Cedar_apple_rust": {

#     "English": {
#         "description": """
#         Cedar apple rust causes yellow or orange spots on leaves.
#         It spreads in humid weather and weakens the plant.
#         """,
#         "cure": """
#         ✅ Apply fungicide in early spring.
#         ✅ Remove infected leaves.
#         ✅ Use resistant varieties.
#         """
#     },

#     "हिंदी": {
#         "description": """
#         सीडर एप्पल रस्ट पत्तियों पर पीले या नारंगी धब्बे बनाता है।
#         यह नम मौसम में तेजी से फैलता है।
#         """,
#         "cure": """
#         ✅ वसंत ऋतु में फंगीसाइड छिड़कें।
#         ✅ संक्रमित पत्तियां हटाएं।
#         ✅ रोग-प्रतिरोधी किस्में लगाएं।
#         """
#     },

#     "मराठी": {
#         "description": """
#         सिडर ॲपल रस्ट पानांवर पिवळे किंवा नारिंगी डाग निर्माण करतो.
#         दमट हवामानात हा रोग पसरतो.
#         """,
#         "cure": """
#         ✅ वसंत ऋतूत फंगीसाइड फवारणी करा.
#         ✅ संक्रमित पाने काढा.
#         ✅ रोगप्रतिकारक वाण वापरा.
#         """
#     }
# },

# # 🍎 Apple Healthy
# "Apple_healthy": {

#     "English": {
#         "description": "The plant is healthy and shows no visible disease symptoms.",
#         "cure": "✅ Maintain proper watering, sunlight, and regular care."
#     },

#     "हिंदी": {
#         "description": "पौधा स्वस्थ है और किसी रोग के लक्षण नहीं दिख रहे हैं।",
#         "cure": "✅ उचित पानी, धूप और नियमित देखभाल बनाए रखें।"
#     },

#     "मराठी": {
#         "description": "झाड पूर्णपणे निरोगी आहे आणि कोणताही रोग नाही.",
#         "cure": "✅ योग्य पाणी, सूर्यप्रकाश आणि काळजी घ्या."
#     }
# },

# # 🍇 Grape Black Rot
# "Grape_Black_rot": {

#     "English": {
#         "description": "Grape black rot causes brown spots on leaves and shriveled fruits.",
#         "cure": "✅ Spray Mancozeb weekly and remove infected leaves."
#     },

#     "हिंदी": {
#         "description": "अंगूर ब्लैक रॉट पत्तियों पर भूरे धब्बे और सूखे फल बनाता है।",
#         "cure": "✅ मैंकोजेब का छिड़काव करें और संक्रमित पत्तियां हटाएं।"
#     },

#     "मराठी": {
#         "description": "द्राक्ष ब्लॅक रॉट पानांवर तपकिरी डाग निर्माण करतो.",
#         "cure": "✅ मॅन्कोझेब फवारणी करा आणि संक्रमित पाने काढा."
#     }
# },

# # 🍇 Grape Esca
# "Grape_Esca_(Black_Measles)": {

#     "English": {
#         "description": "Esca causes leaf discoloration and vine decline.",
#         "cure": "✅ Remove infected vines and avoid water stress."
#     },

#     "हिंदी": {
#         "description": "एस्का रोग पत्तियों का रंग बदल देता है और बेल को कमजोर करता है।",
#         "cure": "✅ संक्रमित बेल हटाएं और पानी का संतुलन रखें।"
#     },

#     "मराठी": {
#         "description": "एस्का रोगामुळे पानांचा रंग बदलतो आणि वेल कमकुवत होते.",
#         "cure": "✅ संक्रमित वेल काढा आणि पाण्याचे व्यवस्थापन करा."
#     }
# },

# # 🍇 Grape Leaf Blight
# "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": {

#     "English": {
#         "description": "Leaf blight causes irregular brown spots and drying of leaves.",
#         "cure": "✅ Apply fungicide and remove infected leaves."
#     },

#     "हिंदी": {
#         "description": "लीफ ब्लाइट पत्तियों पर भूरे धब्बे और सूखापन लाता है।",
#         "cure": "✅ फंगीसाइड छिड़कें और संक्रमित पत्तियां हटाएं।"
#     },

#     "मराठी": {
#         "description": "लीफ ब्लाइट पानांवर तपकिरी डाग आणि कोरडेपणा निर्माण करतो.",
#         "cure": "✅ फंगीसाइड फवारणी करा आणि संक्रमित पाने काढा."
#     }
# },

# # 🍇 Grape Healthy
# "Grape_healthy": {

#     "English": {
#         "description": "The grape plant is healthy with no disease symptoms.",
#         "cure": "✅ Maintain proper sunlight, watering, and pruning."
#     },

#     "हिंदी": {
#         "description": "अंगूर का पौधा स्वस्थ है और कोई रोग नहीं है।",
#         "cure": "✅ उचित धूप, पानी और छंटाई बनाए रखें।"
#     },

#     "मराठी": {
#         "description": "द्राक्षाचे झाड निरोगी आहे आणि कोणताही रोग नाही.",
#         "cure": "✅ योग्य सूर्यप्रकाश, पाणी आणि छाटणी ठेवा."
#     }
# }

# }


# # ---------------- SIDEBAR UPLOAD SECTION ----------------
# with st.sidebar:
#     st.subheader(translations[language]["upload"])

#     options = [
#         translations[language]["upload_image"],
#         translations[language]["use_camera"]
#     ]

#     selected_option = st.radio(
#         translations[language]["choose_source"],
#         options
#     )

#     if selected_option == options[0]:
#         uploaded_file = st.file_uploader(
#             translations[language]["upload_image"],
#             type=["jpg", "jpeg", "png"]
#         )
#     else:
#         uploaded_file = st.camera_input(
#             translations[language]["use_camera"]
#         )



# # ---------------- MAIN CONTENT AREA ----------------
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)

#     col1, col2, col3 = st.columns([1,2,1])
#     with col2:
#         st.image(image, width=450)

#     # Preprocess
#     image = image.convert("RGB")
#     img = tf.image.resize(np.array(image), (224, 224))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)

#     # Prediction
#     prediction = model.predict(img)
#     predicted_index = np.argmax(prediction)
#     predicted_class = class_names[predicted_index]
#     confidence = float(np.max(prediction))

#     if confidence < 0.50:
#         st.warning(translations[language]["low_conf"])
#     else:
#         st.markdown(f"""
#         <div class="result-box">
#             <div class="disease-name">🌱 {predicted_class}</div>
#             <p><strong>Confidence:</strong> {round(confidence * 100, 2)}%</p>
#         </div>
#         """, unsafe_allow_html=True)

#         # Language-based disease info
#         info = disease_info.get(predicted_class)

#         if info and language in info:
#             disease_data = info[language]

#             st.divider()
#             st.subheader(translations[language]["info_title"])

#             st.markdown(f"""
#             <div class="info-box">
#                 <h3>{translations[language]["about"]}</h3>
#                 <p>{disease_data["description"]}</p>
#                 <h3>{translations[language]["cure"]}</h3>
#                 <p>{disease_data["cure"]}</p>
#             </div>
#             """, unsafe_allow_html=True)

#             st.success(translations[language]["complete"])





import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size:50px;
    font-weight:900;
    text-align:center;
    background: linear-gradient(90deg, #1b5e20, #2e7d32, #66bb6a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing:4px;
    margin-bottom:10px;
}

.sub-title {
    font-size:28px;
    text-align:center;
    color:#2e7d32;
    margin-bottom:40px;
}

.disease-name {
    font-size:34px;
    font-weight:bold;
    text-align:center;
    color:#1b5e20;
}

.result-box {
    padding:25px;
    border-radius:12px;
    background-color:#e8f5e9;
    text-align:center;
}

.info-box {
    padding:25px;
    border-radius:12px;
    background-color:#f1f8e9;
    font-size:18px;
    line-height:1.8;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🌿 AI PLANT DISEASE DETECTION</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart Leaf Analysis Using Deep Learning</div>', unsafe_allow_html=True)
st.write("")

# ---------------- LANGUAGE SELECTOR ----------------
language = st.selectbox(
    "🌍 Select Language / भाषा निवडा / ભાષા પસંદ કરો",
    ["English", "हिंदी", "मराठी", "ગુજરાતી"]
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# ---------------- TRANSLATIONS ----------------
translations = {
    "English": {
        "upload": "📷 Upload Leaf Image",
        "choose_source": "Choose Source:",
        "upload_image": "Upload Image",
        "use_camera": "Use Camera",
        "low_conf": "⚠ Low confidence. Please upload a clearer image.",
        "info_title": "📖 Disease Information & Cure",
        "about": "🦠 About the Disease",
        "cure": "💊 Treatment & Prevention",
        "complete": "✅ Analysis Complete",
        "confidence": "Confidence"
    },

    "हिंदी": {
        "upload": "📷 पत्ते की तस्वीर अपलोड करें",
        "choose_source": "स्रोत चुनें:",
        "upload_image": "तस्वीर अपलोड करें",
        "use_camera": "कैमरा उपयोग करें",
        "low_conf": "⚠ कम विश्वसनीयता। कृपया साफ तस्वीर अपलोड करें।",
        "info_title": "📖 रोग की जानकारी और उपचार",
        "about": "🦠 रोग के बारे में",
        "cure": "💊 उपचार और बचाव",
        "complete": "✅ विश्लेषण पूर्ण हुआ",
        "confidence": "विश्वसनीयता"
    },

    "मराठी": {
        "upload": "📷 पानाचा फोटो अपलोड करा",
        "choose_source": "पर्याय निवडा:",
        "upload_image": "फोटो अपलोड करा",
        "use_camera": "कॅमेरा वापरा",
        "low_conf": "⚠ कमी खात्री. कृपया स्पष्ट फोटो अपलोड करा.",
        "info_title": "📖 रोगाची माहिती आणि उपाय",
        "about": "🦠 रोगाबद्दल माहिती",
        "cure": "💊 उपचार आणि प्रतिबंध",
        "complete": "✅ विश्लेषण पूर्ण झाले",
        "confidence": "खात्री"
    },

    "ગુજરાતી": {
        "upload": "📷 પાનનો ફોટો અપલોડ કરો",
        "choose_source": "સ્રોત પસંદ કરો:",
        "upload_image": "ફોટો અપલોડ કરો",
        "use_camera": "કેમેરાનો ઉપયોગ કરો",
        "low_conf": "⚠ ઓછી ખાતરી. કૃપા કરીને વધુ સ્પષ્ટ ફોટો અપલોડ કરો.",
        "info_title": "📖 રોગની માહિતી અને ઉપચાર",
        "about": "🦠 રોગ વિશે માહિતી",
        "cure": "💊 ઉપચાર અને બચાવ",
        "complete": "✅ વિશ્લેષણ પૂર્ણ થયું",
        "confidence": "ખાતરી"
    }
}


# ---------------- MULTI-LANGUAGE DISEASE INFO ----------------
disease_info = {

    # 🍎 Apple Apple Scab
    "Apple_Apple_scab": {

        "English": {
            "description": """
            Apple scab is a fungal disease that causes dark, scabby lesions on leaves and fruits.
            It spreads quickly in cool and wet weather conditions.
            Severe infection can reduce fruit quality and cause early leaf drop.
            """,
            "cure": """
            ✅ Apply fungicides like captan or sulfur.
            ✅ Remove fallen infected leaves.
            ✅ Improve air circulation through pruning.
            ✅ Avoid overhead watering.
            """
        },

        "हिंदी": {
            "description": """
            एप्पल स्कैब एक फंगल रोग है जो पत्तियों और फलों पर काले धब्बे बनाता है।
            यह ठंडे और नम मौसम में तेजी से फैलता है।
            गंभीर संक्रमण से फल की गुणवत्ता कम हो सकती है।
            """,
            "cure": """
            ✅ कैप्टान या सल्फर का छिड़काव करें।
            ✅ गिरी हुई संक्रमित पत्तियां हटाएं।
            ✅ छंटाई करके हवा का प्रवाह बढ़ाएं।
            ✅ ऊपर से पानी देने से बचें।
            """
        },

        "मराठी": {
            "description": """
            ॲपल स्कॅब हा बुरशीजन्य रोग आहे जो पानांवर आणि फळांवर काळे डाग निर्माण करतो.
            थंड आणि दमट हवामानात हा रोग जलद पसरतो.
            जास्त संसर्ग झाल्यास फळांची गुणवत्ता कमी होते.
            """,
            "cure": """
            ✅ कॅप्टान किंवा सल्फरची फवारणी करा.
            ✅ गळलेली संक्रमित पाने काढा.
            ✅ छाटणी करून हवा खेळती ठेवा.
            ✅ वरून पाणी देणे टाळा.
            """
        },

        "ગુજરાતી": {
            "description": """
            એપલ સ્કેબ એક ફૂગજન્ય રોગ છે જે પાન અને ફળ પર કાળા ડાઘ અથવા ખરડિયા જેવા નિશાન બનાવે છે.
            આ રોગ ઠંડા અને ભેજવાળા હવામાનમાં ઝડપથી ફેલાય છે.
            ગંભીર ચેપથી ફળની ગુણવત્તા ઘટી શકે છે અને પાન વહેલા પડી શકે છે.
            """,
            "cure": """
            ✅ કેપ્ટાન અથવા સલ્ફર જેવા ફંગિસાઇડનો છંટકાવ કરો.
            ✅ પડેલા ચેપગ્રસ્ત પાંદડા દૂર કરો.
            ✅ છટણી કરીને હવાની અવરજવર સુધારો.
            ✅ ઉપરથી પાણી આપવાનું ટાળો.
            """
        }
    },

    # 🍎 Apple Black Rot
    "Apple_Black_rot": {

        "English": {
            "description": """
            Black rot causes brown circular spots on leaves and rotting of fruits.
            It spreads in warm and humid weather.
            If untreated, it can severely damage the tree.
            """,
            "cure": """
            ✅ Prune infected branches.
            ✅ Remove infected fruits.
            ✅ Apply copper-based fungicide.
            ✅ Maintain tree health.
            """
        },

        "हिंदी": {
            "description": """
            ब्लैक रॉट पत्तियों पर भूरे धब्बे और फलों में सड़न पैदा करता है।
            यह गर्म और नम मौसम में फैलता है।
            इलाज न करने पर पेड़ को गंभीर नुकसान हो सकता है।
            """,
            "cure": """
            ✅ संक्रमित शाखाएं काटें।
            ✅ संक्रमित फल हटा दें।
            ✅ कॉपर फंगीसाइड का प्रयोग करें।
            ✅ पौधे की उचित देखभाल करें।
            """
        },

        "मराठी": {
            "description": """
            ब्लॅक रॉट पानांवर तपकिरी डाग आणि फळांमध्ये कुज निर्माण करतो.
            उष्ण आणि दमट हवामानात हा रोग वाढतो.
            उपचार न केल्यास झाडाचे मोठे नुकसान होते.
            """,
            "cure": """
            ✅ संक्रमित फांद्या कापून टाका.
            ✅ खराब फळे काढा.
            ✅ कॉपर फंगीसाइड वापरा.
            ✅ झाडाची योग्य निगा ठेवा.
            """
        },

        "ગુજરાતી": {
            "description": """
            બ્લેક રોટ પાન પર ભૂરા ગોળ ડાઘ અને ફળોમાં સડણ પેદા કરે છે.
            આ રોગ ગરમ અને ભેજવાળા હવામાનમાં ફેલાય છે.
            સમયસર ઉપચાર ન કરવાથી ઝાડને ગંભીર નુકસાન થઈ શકે છે.
            """,
            "cure": """
            ✅ ચેપગ્રસ્ત ડાળીઓની છટણી કરો.
            ✅ ચેપગ્રસ્ત ફળો દૂર કરો.
            ✅ કોપર આધારિત ફંગિસાઇડનો ઉપયોગ કરો.
            ✅ ઝાડનું આરોગ્ય જાળવો.
            """
        }
    },

    # 🍎 Apple Cedar Apple Rust
    "Apple_Cedar_apple_rust": {

        "English": {
            "description": """
            Cedar apple rust causes yellow or orange spots on leaves.
            It spreads in humid weather and weakens the plant.
            """,
            "cure": """
            ✅ Apply fungicide in early spring.
            ✅ Remove infected leaves.
            ✅ Use resistant varieties.
            """
        },

        "हिंदी": {
            "description": """
            सीडर एप्पल रस्ट पत्तियों पर पीले या नारंगी धब्बे बनाता है।
            यह नम मौसम में तेजी से फैलता है।
            """,
            "cure": """
            ✅ वसंत ऋतु में फंगीसाइड छिड़कें।
            ✅ संक्रमित पत्तियां हटाएं।
            ✅ रोग-प्रतिरोधी किस्में लगाएं।
            """
        },

        "मराठी": {
            "description": """
            सिडर ॲपल रस्ट पानांवर पिवळे किंवा नारिंगी डाग निर्माण करतो.
            दमट हवामानात हा रोग पसरतो.
            """,
            "cure": """
            ✅ वसंत ऋतूत फंगीसाइड फवारणी करा.
            ✅ संक्रमित पाने काढा.
            ✅ रोगप्रतिकारक वाण वापरा.
            """
        },

        "ગુજરાતી": {
            "description": """
            સીડર એપલ રસ્ટ પાન પર પીળા અથવા નારંગી ડાઘ બનાવે છે.
            આ રોગ ભેજવાળા હવામાનમાં ફેલાય છે અને છોડને નબળો બનાવે છે.
            """,
            "cure": """
            ✅ વસંત ઋતુની શરૂઆતમાં ફંગિસાઇડનો છંટકાવ કરો.
            ✅ ચેપગ્રસ્ત પાંદડા દૂર કરો.
            ✅ રોગપ્રતિકારક જાતો વાવો.
            """
        }
    },

    # 🍎 Apple Healthy
    "Apple_healthy": {

        "English": {
            "description": "The plant is healthy and shows no visible disease symptoms.",
            "cure": "✅ Maintain proper watering, sunlight, and regular care."
        },

        "हिंदी": {
            "description": "पौधा स्वस्थ है और किसी रोग के लक्षण नहीं दिख रहे हैं।",
            "cure": "✅ उचित पानी, धूप और नियमित देखभाल बनाए रखें।"
        },

        "मराठी": {
            "description": "झाड पूर्णपणे निरोगी आहे आणि कोणताही रोग नाही.",
            "cure": "✅ योग्य पाणी, सूर्यप्रकाश आणि काळजी घ्या."
        },

        "ગુજરાતી": {
            "description": "છોડ સ્વસ્થ છે અને કોઈ દેખાતા રોગના લક્ષણો નથી.",
            "cure": "✅ યોગ્ય પાણી, સૂર્યપ્રકાશ અને નિયમિત કાળજી જાળવો."
        }
    },

    # 🍇 Grape Black Rot
    "Grape_Black_rot": {

        "English": {
            "description": "Grape black rot causes brown spots on leaves and shriveled fruits.",
            "cure": "✅ Spray Mancozeb weekly and remove infected leaves."
        },

        "हिंदी": {
            "description": "अंगूर ब्लैक रॉट पत्तियों पर भूरे धब्बे और सूखे फल बनाता है।",
            "cure": "✅ मैंकोजेब का छिड़काव करें और संक्रमित पत्तियां हटाएं।"
        },

        "मराठी": {
            "description": "द्राक्ष ब्लॅक रॉट पानांवर तपकिरी डाग निर्माण करतो.",
            "cure": "✅ मॅन्कोझेब फवारणी करा आणि संक्रमित पाने काढा."
        },

        "ગુજરાતી": {
            "description": "દ્રાક્ષનો બ્લેક રોટ પાન પર ભૂરા ડાઘ અને સુકાઈ ગયેલા ફળો પેદા કરે છે.",
            "cure": "✅ દર અઠવાડિયે મેન્કોઝેબનો છંટકાવ કરો અને ચેપગ્રસ્ત પાંદડા દૂર કરો."
        }
    },

    # 🍇 Grape Esca
    "Grape_Esca_(Black_Measles)": {

        "English": {
            "description": "Esca causes leaf discoloration and vine decline.",
            "cure": "✅ Remove infected vines and avoid water stress."
        },

        "हिंदी": {
            "description": "एस्का रोग पत्तियों का रंग बदल देता है और बेल को कमजोर करता है।",
            "cure": "✅ संक्रमित बेल हटाएं और पानी का संतुलन रखें।"
        },

        "मराठी": {
            "description": "एस्का रोगामुळे पानांचा रंग बदलतो आणि वेल कमकुवत होते.",
            "cure": "✅ संक्रमित वेल काढा आणि पाण्याचे व्यवस्थापन करा."
        },

        "ગુજરાતી": {
            "description": "એસ્કા રોગ પાનનો રંગ બદલી નાખે છે અને વેલને નબળી બનાવે છે.",
            "cure": "✅ ચેપગ્રસ્ત વેલો દૂર કરો અને પાણીનો તણાવ ટાળો."
        }
    },

    # 🍇 Grape Leaf Blight
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": {

        "English": {
            "description": "Leaf blight causes irregular brown spots and drying of leaves.",
            "cure": "✅ Apply fungicide and remove infected leaves."
        },

        "हिंदी": {
            "description": "लीफ ब्लाइट पत्तियों पर भूरे धब्बे और सूखापन लाता है।",
            "cure": "✅ फंगीसाइड छिड़कें और संक्रमित पत्तियां हटाएं।"
        },

        "मराठी": {
            "description": "लीफ ब्लाइट पानांवर तपकिरी डाग आणि कोरडेपणा निर्माण करतो.",
            "cure": "✅ फंगीसाइड फवारणी करा आणि संक्रमित पाने काढा."
        },

        "ગુજરાતી": {
            "description": "લીફ બ્લાઇટ પાન પર અનિયમિત ભૂરા ડાઘ અને પાન સુકાવાનું કારણ બને છે.",
            "cure": "✅ ફંગિસાઇડનો છંટકાવ કરો અને ચેપગ્રસ્ત પાંદડા દૂર કરો."
        }
    },

    # 🍇 Grape Healthy
    "Grape_healthy": {

        "English": {
            "description": "The grape plant is healthy with no disease symptoms.",
            "cure": "✅ Maintain proper sunlight, watering, and pruning."
        },

        "हिंदी": {
            "description": "अंगूर का पौधा स्वस्थ है और कोई रोग नहीं है।",
            "cure": "✅ उचित धूप, पानी और छंटाई बनाए रखें।"
        },

        "मराठी": {
            "description": "द्राक्षाचे झाड निरोगी आहे आणि कोणताही रोग नाही.",
            "cure": "✅ योग्य सूर्यप्रकाश, पाणी आणि छाटणी ठेवा."
        },

        "ગુજરાતી": {
            "description": "દ્રાક્ષનો છોડ સ્વસ્થ છે અને કોઈ રોગના લક્ષણો નથી.",
            "cure": "✅ યોગ્ય સૂર્યપ્રકાશ, પાણી અને છટણી જાળવો."
        }
    }

}


# ---------------- SIDEBAR UPLOAD SECTION ----------------
with st.sidebar:
    st.subheader(translations[language]["upload"])

    options = [
        translations[language]["upload_image"],
        translations[language]["use_camera"]
    ]

    selected_option = st.radio(
        translations[language]["choose_source"],
        options
    )

    if selected_option == options[0]:
        uploaded_file = st.file_uploader(
            translations[language]["upload_image"],
            type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_file = st.camera_input(
            translations[language]["use_camera"]
        )


# ---------------- MAIN CONTENT AREA ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=450)

    # ---------------- PREPROCESS IMAGE ----------------
    image = image.convert("RGB")
    img = tf.image.resize(np.array(image), (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ---------------- PREDICTION ----------------
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))

    if confidence < 0.50:
        st.warning(translations[language]["low_conf"])

    else:
        st.markdown(f"""
        <div class="result-box">
            <div class="disease-name">🌱 {predicted_class}</div>
            <p><strong>{translations[language]["confidence"]}:</strong> {round(confidence * 100, 2)}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- LANGUAGE-BASED DISEASE INFO ----------------
        info = disease_info.get(predicted_class)

        if info and language in info:
            disease_data = info[language]

            st.divider()
            st.subheader(translations[language]["info_title"])

            st.markdown(f"""
            <div class="info-box">
                <h3>{translations[language]["about"]}</h3>
                <p>{disease_data["description"]}</p>
                <h3>{translations[language]["cure"]}</h3>
                <p>{disease_data["cure"]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.success(translations[language]["complete"])

        else:
            st.warning("Disease information is not available for this class/language.")



