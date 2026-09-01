
import os
import json
import numpy as np
import onnxruntime as ort

from PIL import Image
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session
)
from werkzeug.utils import secure_filename


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = "agrozyen-secret-key"


# =========================================================
# TRANSLATIONS
# =========================================================

TRANSLATIONS = {

    # =====================================================
    # ENGLISH
    # =====================================================

    "en": {

        # NAVIGATION
        "home": "Home",
        "detect": "Detect Disease",
        "performance": "Performance",
        "about": "About",
        "get_started": "Get Started",

        # HOME
        "tagline": "🌿 AI-POWERED AGRICULTURE",
        "hero_title": "Protect Your Crops",
        "hero_ai": "With AI",

        "description":
            "Detect crop diseases quickly using artificial intelligence. "
            "Upload a leaf image and get an instant disease prediction.",

        "detect_button": "🔍 Detect Disease",
        "how_button": "How It Works →",

        "ai_detection": "AI Disease Detection",

        "ai_description":
            "Upload a crop leaf image and let AI analyze it.",

        "upload": "Upload Image",

        "upload_text":
            "Upload a clear image of your crop leaf.",

        "analysis": "AI Analysis",

        "analysis_text":
            "Our AI model analyzes the leaf for disease.",

        "results": "Get Results",

        "results_text":
            "Receive disease prediction and confidence.",

        "simple_process": "SIMPLE PROCESS",
        "how_it_works": "How It Works",

        "steps_description":
            "Detecting crop diseases takes only a few simple steps.",

        "step_upload": "Upload",

        "step_upload_text":
            "Upload a photo of the affected crop leaf.",

        "step_analyze": "Analyze",

        "step_analyze_text":
            "AI analyzes visual patterns in the leaf.",

        "step_detect": "Detect",

        "step_detect_text":
            "The system identifies the possible disease.",

        "step_learn": "Learn",

        "step_learn_text":
            "View disease information and prevention tips.",

        # PERFORMANCE
        "performance_tagline": "AI MODEL EVALUATION",

        "performance_title": "Model Performance",

        "performance_description":
            "Performance of the AgroZyen AI disease classification "
            "model on the validation dataset.",

        "validation_accuracy": "Validation Accuracy",

        "overall_accuracy":
            "Overall model accuracy",

        "correct_predictions": "Correct Predictions",

        "correct_predictions_text":
            "Correctly classified images",

        "total_classes": "Total Classes",

        "total_classes_text":
            "Crop disease categories",

        "per_class_accuracy": "Per-Class Accuracy",

        "per_class_description":
            "Accuracy achieved for each crop disease category.",

        "corn_common_rust": "Corn Common Rust",
        "corn_northern_leaf_blight": "Corn Northern Leaf Blight",
        "corn_healthy": "Corn Healthy",

        "potato_early_blight": "Potato Early Blight",
        "potato_late_blight": "Potato Late Blight",
        "potato_healthy": "Potato Healthy",

        "tomato_early_blight": "Tomato Early Blight",
        "tomato_late_blight": "Tomato Late Blight",
        "tomato_healthy": "Tomato Healthy",

        "confusion_matrix": "Confusion Matrix",

        "confusion_description":
            "The confusion matrix shows actual classes compared "
            "with predicted classes.",

        "actual_predicted": "Actual / Predicted",

        "corn_rust": "Corn Rust",
        "corn_blight": "Corn Blight",
        "corn_healthy": "Corn Healthy",

        "potato_early": "Potato Early",
        "potato_late": "Potato Late",
        "potato_healthy": "Potato Healthy",

        "tomato_early": "Tomato Early",
        "tomato_late": "Tomato Late",
        "tomato_healthy": "Tomato Healthy",

        "try_detection": "🔍 Try Disease Detection",
        "back_home": "🏠 Back to Home",

        # ABOUT
        "about_tagline": "ABOUT AGROZYEN AI",

        "about_title": "Smart Agriculture With AI",

        "about_description":
            "AgroZyen AI is an AI-powered crop disease detection "
            "system designed to help identify crop diseases quickly "
            "using leaf images.",

        "about_project_title": "About AgroZyen",

        "about_project_text":
            "AgroZyen AI helps farmers identify crop diseases quickly "
            "by analyzing images of crop leaves using artificial intelligence.",

        "about_ai_title": "Artificial Intelligence",

        "about_ai_text":
            "Our AI model analyzes visual patterns in crop leaves "
            "and predicts the most likely disease category.",

        "about_farmer_title": "Built For Farmers",

        "about_farmer_text":
            "AgroZyen AI provides a simple and accessible way for "
            "farmers to check crop health and receive useful disease information.",

        "about_process_tag": "HOW AGROZYEN WORKS",

        "about_process_title": "Simple AI-Powered Process",

        "about_process_description":
            "AgroZyen AI uses a simple three-step process to analyze "
            "crop leaves and identify possible diseases.",

        "about_step1_title": "Upload Image",

        "about_step1_text":
            "Upload a clear image of the crop leaf you want to analyze.",

        "about_step2_title": "AI Analysis",

        "about_step2_text":
            "The trained AI model analyzes the visual features of the uploaded leaf.",

        "about_step3_title": "Get Prediction",

        "about_step3_text":
            "The system provides the predicted disease, confidence level, "
            "and useful information.",

        "about_technology_title": "AI Technology",

        "about_technology_text":
            "AgroZyen AI uses a trained deep learning image classification "
            "model to recognize crop disease patterns.",

        "about_accuracy_title": "Model Performance",

        "about_accuracy_text":
            "The model is evaluated using validation data across nine crop "
            "health and disease categories.",

        "about_cta_title": "Ready To Check Your Crop?",

        "about_cta_text":
            "Upload a crop leaf image and let AgroZyen AI analyze it.",

        # DETECT
        "detect_tagline": "AI DISEASE DETECTION",

        "detect_title": "Detect Crop Disease",

        "detect_description":
            "Upload an image of a crop leaf and our AI model will "
            "analyze it to identify possible diseases.",

        "select_image": "Select Crop Leaf Image",

        "choose_image": "Choose Image",

        "analyze_image": "🔍 Analyze Image",

        "supported_formats":
            "Supported formats: JPG, JPEG, PNG",

        # RESULT
        "result_tagline": "AI DETECTION RESULT",

        "result_title": "Disease Detection Result",

        "prediction": "Prediction",

        "disease_detected": "Disease Detected",

        "healthy_crop": "Healthy Crop",

        "crop": "Crop",

        "confidence": "Confidence",

        "confidence_level": "Confidence Level",

        "high": "High",

        "moderate": "Moderate",

        "low": "Low",

        "about_disease": "About This Condition",

        "symptoms": "Common Symptoms",

        "recommendation": "Recommendation",

        "no_symptoms": "No disease symptoms detected.",

        "analyze_another": "🔍 Analyze Another Image",

        "return_home": "🏠 Return Home",

        "prediction_error":
            "Error while analyzing the image.",

        # INVALID IMAGE
        "invalid_image":
            "Please upload a clear Corn, Potato, or Tomato leaf image.",

        "invalid_image_title":
            "Invalid Image",

        "invalid_image_description":
            "The uploaded image does not appear to be a suitable crop leaf image. "
            "Please upload a clear image of a Corn, Potato, or Tomato leaf.",

        # FOOTER
        "footer":
            "AI-powered crop disease detection system."
    },


    # =====================================================
    # HINDI
    # =====================================================

    "hi": {

        "home": "होम",
        "detect": "रोग पहचानें",
        "performance": "प्रदर्शन",
        "about": "हमारे बारे में",
        "get_started": "शुरू करें",

        "tagline": "🌿 AI-आधारित कृषि",
        "hero_title": "अपनी फसलों की रक्षा करें",
        "hero_ai": "AI के साथ",

        "description":
            "कृत्रिम बुद्धिमत्ता का उपयोग करके फसल रोगों की जल्दी पहचान करें। "
            "पत्ते की तस्वीर अपलोड करें और तुरंत रोग की भविष्यवाणी प्राप्त करें।",

        "detect_button": "🔍 रोग पहचानें",
        "how_button": "यह कैसे काम करता है →",

        "ai_detection": "AI रोग पहचान",

        "ai_description":
            "फसल के पत्ते की तस्वीर अपलोड करें और AI को उसका विश्लेषण करने दें।",

        "upload": "तस्वीर अपलोड करें",

        "upload_text":
            "अपने फसल के पत्ते की साफ तस्वीर अपलोड करें।",

        "analysis": "AI विश्लेषण",

        "analysis_text":
            "हमारा AI मॉडल पत्ते का रोग के लिए विश्लेषण करता है।",

        "results": "परिणाम प्राप्त करें",

        "results_text":
            "रोग की भविष्यवाणी और विश्वास स्तर प्राप्त करें।",

        "simple_process": "सरल प्रक्रिया",
        "how_it_works": "यह कैसे काम करता है",

        "steps_description":
            "फसल रोग की पहचान केवल कुछ आसान चरणों में करें।",

        "step_upload": "अपलोड",

        "step_upload_text":
            "प्रभावित फसल के पत्ते की तस्वीर अपलोड करें।",

        "step_analyze": "विश्लेषण",

        "step_analyze_text":
            "AI पत्ते के दृश्य पैटर्न का विश्लेषण करता है।",

        "step_detect": "पहचान",

        "step_detect_text":
            "सिस्टम संभावित रोग की पहचान करता है।",

        "step_learn": "जानकारी",

        "step_learn_text":
            "रोग की जानकारी और बचाव के उपाय देखें.",

        "performance_tagline": "AI मॉडल मूल्यांकन",
        "performance_title": "मॉडल का प्रदर्शन",

        "performance_description":
            "मान्यता डेटासेट पर AgroZyen AI रोग वर्गीकरण मॉडल का प्रदर्शन।",

        "validation_accuracy": "मान्यता सटीकता",
        "overall_accuracy": "मॉडल की कुल सटीकता",

        "correct_predictions": "सही भविष्यवाणियाँ",

        "correct_predictions_text":
            "सही तरीके से वर्गीकृत की गई तस्वीरें",

        "total_classes": "कुल श्रेणियाँ",

        "total_classes_text":
            "फसल रोग की श्रेणियाँ",

        "per_class_accuracy": "प्रति-श्रेणी सटीकता",

        "per_class_description":
            "प्रत्येक फसल रोग श्रेणी के लिए प्राप्त सटीकता।",

        "corn_common_rust": "मक्का सामान्य रतुआ",
        "corn_northern_leaf_blight": "मक्का उत्तरी पत्ती झुलसा रोग",
        "corn_healthy": "स्वस्थ मक्का",

        "potato_early_blight": "आलू अर्ली ब्लाइट",
        "potato_late_blight": "आलू लेट ब्लाइट",
        "potato_healthy": "स्वस्थ आलू",

        "tomato_early_blight": "टमाटर अर्ली ब्लाइट",
        "tomato_late_blight": "टमाटर लेट ब्लाइट",
        "tomato_healthy": "स्वस्थ टमाटर",

        "confusion_matrix": "कन्फ्यूजन मैट्रिक्स",

        "confusion_description":
            "कन्फ्यूजन मैट्रिक्स वास्तविक श्रेणियों की तुलना "
            "अनुमानित श्रेणियों से करता है।",

        "actual_predicted": "वास्तविक / अनुमानित",

        "corn_rust": "मक्का रतुआ",
        "corn_blight": "मक्का झुलसा रोग",
        "corn_healthy": "स्वस्थ मक्का",

        "potato_early": "आलू अर्ली",
        "potato_late": "आलू लेट",
        "potato_healthy": "स्वस्थ आलू",

        "tomato_early": "टमाटर अर्ली",
        "tomato_late": "टमाटर लेट",
        "tomato_healthy": "स्वस्थ टमाटर",

        "try_detection": "🔍 रोग पहचानने का प्रयास करें",
        "back_home": "🏠 होम पर वापस जाएँ",

        "about_tagline": "AGROZYEN AI के बारे में",
        "about_title": "AI के साथ स्मार्ट कृषि",

        "about_description":
            "AgroZyen AI एक AI-आधारित फसल रोग पहचान प्रणाली है "
            "जो पत्तियों की तस्वीरों का उपयोग करके फसल रोगों की जल्दी पहचान करने में मदद करती है।",

        "about_project_title": "AgroZyen के बारे में",

        "about_project_text":
            "AgroZyen AI कृत्रिम बुद्धिमत्ता का उपयोग करके फसल की पत्तियों "
            "की तस्वीरों का विश्लेषण करता है और किसानों को रोगों की पहचान करने में मदद करता है।",

        "about_ai_title": "कृत्रिम बुद्धिमत्ता",

        "about_ai_text":
            "हमारा AI मॉडल फसल की पत्तियों में मौजूद दृश्य पैटर्न का "
            "विश्लेषण करके संभावित रोग की श्रेणी बताता है।",

        "about_farmer_title": "किसानों के लिए बनाया गया",

        "about_farmer_text":
            "AgroZyen AI किसानों को फसल के स्वास्थ्य की जांच करने "
            "और उपयोगी रोग संबंधी जानकारी प्राप्त करने का सरल तरीका प्रदान करता है।",

        "about_process_tag": "AGROZYEN कैसे काम करता है",

        "about_process_title": "सरल AI-आधारित प्रक्रिया",

        "about_process_description":
            "AgroZyen AI फसल की पत्तियों का विश्लेषण करने और संभावित रोग "
            "की पहचान करने के लिए तीन सरल चरणों का उपयोग करता है।",

        "about_step1_title": "तस्वीर अपलोड करें",

        "about_step1_text":
            "जिस फसल के पत्ते का विश्लेषण करना है उसकी साफ तस्वीर अपलोड करें।",

        "about_step2_title": "AI विश्लेषण",

        "about_step2_text":
            "प्रशिक्षित AI मॉडल अपलोड किए गए पत्ते की दृश्य विशेषताओं का विश्लेषण करता है।",

        "about_step3_title": "भविष्यवाणी प्राप्त करें",

        "about_step3_text":
            "सिस्टम संभावित रोग, विश्वास स्तर और उपयोगी जानकारी प्रदान करता है।",

        "about_technology_title": "AI तकनीक",

        "about_technology_text":
            "AgroZyen AI फसल रोगों के पैटर्न को पहचानने के लिए "
            "प्रशिक्षित डीप लर्निंग इमेज क्लासिफिकेशन मॉडल का उपयोग करता है।",

        "about_accuracy_title": "मॉडल का प्रदर्शन",

        "about_accuracy_text":
            "मॉडल का मूल्यांकन नौ फसल स्वास्थ्य और रोग श्रेणियों के "
            "मान्यता डेटा का उपयोग करके किया गया है।",

        "about_cta_title": "क्या आप अपनी फसल की जांच करना चाहते हैं?",

        "about_cta_text":
            "फसल के पत्ते की तस्वीर अपलोड करें और AgroZyen AI को उसका विश्लेषण करने दें।",

        "detect_tagline": "AI रोग पहचान",

        "detect_title": "फसल रोग पहचानें",

        "detect_description":
            "फसल के पत्ते की तस्वीर अपलोड करें और हमारा AI मॉडल "
            "उसका विश्लेषण करके संभावित रोग की पहचान करेगा।",

        "select_image": "फसल के पत्ते की तस्वीर चुनें",
        "choose_image": "तस्वीर चुनें",
        "analyze_image": "🔍 तस्वीर का विश्लेषण करें",

        "supported_formats":
            "समर्थित प्रारूप: JPG, JPEG, PNG",

        "result_tagline": "AI पहचान परिणाम",
        "result_title": "रोग पहचान परिणाम",

        "prediction": "भविष्यवाणी",
        "disease_detected": "रोग की पहचान",
        "healthy_crop": "स्वस्थ फसल",

        "crop": "फसल",
        "confidence": "विश्वास स्तर",
        "confidence_level": "विश्वास स्तर",

        "high": "उच्च",
        "moderate": "मध्यम",
        "low": "कम",

        "about_disease": "इस स्थिति के बारे में",
        "symptoms": "सामान्य लक्षण",
        "recommendation": "सुझाव",

        "no_symptoms": "रोग के कोई लक्षण नहीं पाए गए।",

        "analyze_another": "🔍 दूसरी तस्वीर का विश्लेषण करें",
        "return_home": "🏠 होम पर वापस जाएँ",

        "prediction_error":
            "तस्वीर का विश्लेषण करते समय त्रुटि हुई।",

        "invalid_image":
            "कृपया मक्का, आलू या टमाटर के पत्ते की साफ तस्वीर अपलोड करें।",

        "invalid_image_title":
            "अमान्य तस्वीर",

        "invalid_image_description":
            "अपलोड की गई तस्वीर उपयुक्त फसल के पत्ते की तस्वीर नहीं लगती। "
            "कृपया मक्का, आलू या टमाटर के पत्ते की साफ तस्वीर अपलोड करें।",

        "footer":
            "AI आधारित फसल रोग पहचान प्रणाली।"
    },


    # =====================================================
    # MARATHI
    # =====================================================

    "mr": {

        "home": "मुख्यपृष्ठ",
        "detect": "रोग ओळखा",
        "performance": "कामगिरी",
        "about": "आमच्याबद्दल",
        "get_started": "सुरुवात करा",

        "tagline": "🌿 AI-आधारित शेती",
        "hero_title": "तुमच्या पिकांचे संरक्षण करा",
        "hero_ai": "AI च्या मदतीने",

        "description":
            "कृत्रिम बुद्धिमत्तेचा वापर करून पिकांचे रोग लवकर ओळखा. "
            "पानाचा फोटो अपलोड करा आणि त्वरित रोगाची माहिती मिळवा.",

        "detect_button": "🔍 रोग ओळखा",
        "how_button": "हे कसे कार्य करते →",

        "ai_detection": "AI रोग ओळख",

        "ai_description":
            "पिकाच्या पानाचा फोटो अपलोड करा आणि AI ला त्याचे विश्लेषण करू द्या.",

        "upload": "प्रतिमा अपलोड करा",

        "upload_text":
            "तुमच्या पिकाच्या पानाचा स्पष्ट फोटो अपलोड करा.",

        "analysis": "AI विश्लेषण",

        "analysis_text":
            "आमचे AI मॉडेल पानाचे रोगासाठी विश्लेषण करते.",

        "results": "निकाल मिळवा",

        "results_text":
            "रोगाची माहिती आणि विश्वास पातळी मिळवा.",

        "simple_process": "सोप्पी प्रक्रिया",
        "how_it_works": "हे कसे कार्य करते",

        "steps_description":
            "पिकांचे रोग ओळखण्यासाठी फक्त काही सोप्या चरणांची आवश्यकता आहे.",

        "step_upload": "अपलोड",

        "step_upload_text":
            "प्रभावित पिकाच्या पानाचा फोटो अपलोड करा.",

        "step_analyze": "विश्लेषण",

        "step_analyze_text":
            "AI पानातील दृश्य नमुन्यांचे विश्लेषण करते.",

        "step_detect": "ओळख",

        "step_detect_text":
            "सिस्टम संभाव्य रोग ओळखते.",

        "step_learn": "शिका",

        "step_learn_text":
            "रोगाची माहिती आणि प्रतिबंधक उपाय पहा.",

        "performance_tagline": "AI मॉडेल मूल्यांकन",

        "performance_title": "मॉडेलची कामगिरी",

        "performance_description":
            "मान्यता डेटासेटवर AgroZyen AI रोग वर्गीकरण मॉडेलची कामगिरी.",

        "validation_accuracy": "मान्यता अचूकता",

        "overall_accuracy": "मॉडेलची एकूण अचूकता",

        "correct_predictions": "योग्य अंदाज",

        "correct_predictions_text":
            "योग्यरित्या वर्गीकृत केलेल्या प्रतिमा",

        "total_classes": "एकूण वर्ग",

        "total_classes_text":
            "पीक रोगांच्या श्रेणी",

        "per_class_accuracy": "प्रति-वर्ग अचूकता",

        "per_class_description":
            "प्रत्येक पीक रोग श्रेणीसाठी मिळालेली अचूकता.",

        "corn_common_rust": "मका सामान्य गंज",
        "corn_northern_leaf_blight": "मका नॉर्दर्न लीफ ब्लाइट",
        "corn_healthy": "निरोगी मका",

        "potato_early_blight": "बटाटा अर्ली ब्लाइट",
        "potato_late_blight": "बटाटा लेट ब्लाइट",
        "potato_healthy": "निरोगी बटाटा",

        "tomato_early_blight": "टोमॅटो अर्ली ब्लाइट",
        "tomato_late_blight": "टोमॅटो लेट ब्लाइट",
        "tomato_healthy": "निरोगी टोमॅटो",

        "confusion_matrix": "कन्फ्यूजन मॅट्रिक्स",

        "confusion_description":
            "कन्फ्यूजन मॅट्रिक्स वास्तविक वर्गांची तुलना अंदाज केलेल्या वर्गांशी करते.",

        "actual_predicted": "वास्तविक / अंदाजित",

        "corn_rust": "मका गंज",
        "corn_blight": "मका ब्लाइट",
        "corn_healthy": "निरोगी मका",

        "potato_early": "बटाटा अर्ली",
        "potato_late": "बटाटा लेट",
        "potato_healthy": "निरोगी बटाटा",

        "tomato_early": "टोमॅटो अर्ली",
        "tomato_late": "टोमॅटो लेट",
        "tomato_healthy": "निरोगी टोमॅटो",

        "try_detection": "🔍 रोग ओळखण्याचा प्रयत्न करा",
        "back_home": "🏠 मुख्यपृष्ठावर परत जा",

        "about_tagline": "AGROZYEN AI बद्दल",

        "about_title": "AI सह स्मार्ट शेती",

        "about_description":
            "AgroZyen AI ही AI-आधारित पीक रोग ओळख प्रणाली आहे "
            "जी पानांच्या प्रतिमांचा वापर करून पीक रोग लवकर ओळखण्यास मदत करते.",

        "about_project_title": "AgroZyen बद्दल",

        "about_project_text":
            "AgroZyen AI कृत्रिम बुद्धिमत्तेचा वापर करून पिकांच्या "
            "पानांच्या प्रतिमांचे विश्लेषण करते आणि शेतकऱ्यांना रोग ओळखण्यास मदत करते.",

        "about_ai_title": "कृत्रिम बुद्धिमत्ता",

        "about_ai_text":
            "आमचे AI मॉडेल पिकांच्या पानांमधील दृश्य नमुन्यांचे "
            "विश्लेषण करून संभाव्य रोगाची श्रेणी सांगते.",

        "about_farmer_title": "शेतकऱ्यांसाठी तयार केलेले",

        "about_farmer_text":
            "AgroZyen AI शेतकऱ्यांना पिकांच्या आरोग्याची तपासणी "
            "करण्यासाठी आणि रोगाची उपयुक्त माहिती मिळवण्यासाठी सोपा मार्ग देते.",

        "about_process_tag": "AGROZYEN कसे कार्य करते",

        "about_process_title": "सोप्पी AI-आधारित प्रक्रिया",

        "about_process_description":
            "AgroZyen AI पिकांच्या पानांचे विश्लेषण करण्यासाठी "
            "आणि संभाव्य रोग ओळखण्यासाठी तीन सोप्या चरणांचा वापर करते.",

        "about_step1_title": "प्रतिमा अपलोड करा",

        "about_step1_text":
            "ज्या पिकाच्या पानाचे विश्लेषण करायचे आहे त्याचा स्पष्ट फोटो अपलोड करा.",

        "about_step2_title": "AI विश्लेषण",

        "about_step2_text":
            "प्रशिक्षित AI मॉडेल अपलोड केलेल्या पानाच्या दृश्य वैशिष्ट्यांचे विश्लेषण करते.",

        "about_step3_title": "अंदाज मिळवा",

        "about_step3_text":
            "सिस्टम संभाव्य रोग, विश्वास पातळी आणि उपयुक्त माहिती प्रदान करते.",

        "about_technology_title": "AI तंत्रज्ञान",

        "about_technology_text":
            "AgroZyen AI पीक रोगांचे नमुने ओळखण्यासाठी "
            "प्रशिक्षित डीप लर्निंग इमेज क्लासिफिकेशन मॉडेलचा वापर करते.",

        "about_accuracy_title": "मॉडेलची कामगिरी",

        "about_accuracy_text":
            "मॉडेलचे मूल्यांकन नऊ पीक आरोग्य आणि रोग श्रेणींच्या "
            "मान्यता डेटाचा वापर करून केले जाते.",

        "about_cta_title": "तुमचे पीक तपासण्यासाठी तयार आहात?",

        "about_cta_text":
            "पिकाच्या पानाची प्रतिमा अपलोड करा आणि AgroZyen AI ला तिचे विश्लेषण करू द्या.",

        "detect_tagline": "AI रोग ओळख",

        "detect_title": "पीक रोग ओळखा",

        "detect_description":
            "पिकाच्या पानाची प्रतिमा अपलोड करा आणि आमचे AI मॉडेल "
            "तिचे विश्लेषण करून संभाव्य रोग ओळखेल.",

        "select_image": "पिकाच्या पानाची प्रतिमा निवडा",

        "choose_image": "प्रतिमा निवडा",

        "analyze_image": "🔍 प्रतिमेचे विश्लेषण करा",

        "supported_formats":
            "समर्थित स्वरूप: JPG, JPEG, PNG",

        "result_tagline": "AI ओळख परिणाम",

        "result_title": "रोग ओळख परिणाम",

        "prediction": "अंदाज",

        "disease_detected": "ओळखलेला रोग",

        "healthy_crop": "निरोगी पीक",

        "crop": "पीक",

        "confidence": "विश्वास",

        "confidence_level": "विश्वास पातळी",

        "high": "उच्च",

        "moderate": "मध्यम",

        "low": "कमी",

        "about_disease": "या स्थितीबद्दल",

        "symptoms": "सामान्य लक्षणे",

        "recommendation": "शिफारस",

        "no_symptoms": "रोगाची कोणतीही लक्षणे आढळली नाहीत.",

        "analyze_another": "🔍 दुसऱ्या प्रतिमेचे विश्लेषण करा",

        "return_home": "🏠 मुख्यपृष्ठावर परत जा",

        "prediction_error":
            "प्रतिमेचे विश्लेषण करताना त्रुटी आली.",

        "invalid_image":
            "कृपया मका, बटाटा किंवा टोमॅटोच्या पानाचा स्पष्ट फोटो अपलोड करा.",

        "invalid_image_title":
            "अवैध प्रतिमा",

        "invalid_image_description":
            "अपलोड केलेली प्रतिमा योग्य पिकाच्या पानाची प्रतिमा दिसत नाही. "
            "कृपया मका, बटाटा किंवा टोमॅटोच्या पानाचा स्पष्ट फोटो अपलोड करा.",

        "footer":
            "AI-आधारित पीक रोग ओळख प्रणाली."
    },


    # =====================================================
    # TELUGU
    # =====================================================

    "te": {

        "home": "హోమ్",
        "detect": "వ్యాధిని గుర్తించండి",
        "performance": "పనితీరు",
        "about": "మా గురించి",
        "get_started": "ప్రారంభించండి",

        "tagline": "🌿 AI ఆధారిత వ్యవసాయం",
        "hero_title": "మీ పంటలను రక్షించండి",
        "hero_ai": "AI తో",

        "description":
            "కృత్రిమ మేధస్సును ఉపయోగించి పంట వ్యాధులను త్వరగా గుర్తించండి. "
            "ఆకు చిత్రాన్ని అప్‌లోడ్ చేసి వెంటనే వ్యాధి అంచనాను పొందండి.",

        "detect_button": "🔍 వ్యాధిని గుర్తించండి",
        "how_button": "ఇది ఎలా పనిచేస్తుంది →",

        "ai_detection": "AI వ్యాధి గుర్తింపు",

        "ai_description":
            "పంట ఆకు చిత్రాన్ని అప్‌లోడ్ చేసి AI దాన్ని విశ్లేషించనివ్వండి.",

        "upload": "చిత్రాన్ని అప్‌లోడ్ చేయండి",

        "upload_text":
            "మీ పంట ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి.",

        "analysis": "AI విశ్లేషణ",

        "analysis_text":
            "మా AI మోడల్ ఆకును వ్యాధి కోసం విశ్లేషిస్తుంది.",

        "results": "ఫలితాలను పొందండి",

        "results_text":
            "వ్యాధి అంచనా మరియు నమ్మక స్థాయిని పొందండి.",

        "simple_process": "సులభమైన ప్రక్రియ",
        "how_it_works": "ఇది ఎలా పనిచేస్తుంది",

        "steps_description":
            "పంట వ్యాధులను గుర్తించడానికి కొన్ని సులభమైన దశలు మాత్రమే అవసరం.",

        "step_upload": "అప్‌లోడ్",

        "step_upload_text":
            "ప్రభావిత పంట ఆకు ఫోటోను అప్‌లోడ్ చేయండి.",

        "step_analyze": "విశ్లేషించండి",

        "step_analyze_text":
            "AI ఆకులోని దృశ్య నమూనాలను విశ్లేషిస్తుంది.",

        "step_detect": "గుర్తించండి",

        "step_detect_text":
            "సిస్టమ్ సంభావ్య వ్యాధిని గుర్తిస్తుంది.",

        "step_learn": "తెలుసుకోండి",

        "step_learn_text":
            "వ్యాధి సమాచారం మరియు నివారణ చిట్కాలను చూడండి.",

        "performance_tagline": "AI మోడల్ మూల్యాంకనం",

        "performance_title": "మోడల్ పనితీరు",

        "performance_description":
            "ధృవీకరణ డేటాసెట్‌లో AgroZyen AI వ్యాధి వర్గీకరణ మోడల్ పనితీరు.",

        "validation_accuracy": "ధృవీకరణ ఖచ్చితత్వం",

        "overall_accuracy": "మొత్తం మోడల్ ఖచ్చితత్వం",

        "correct_predictions": "సరైన అంచనాలు",

        "correct_predictions_text":
            "సరిగ్గా వర్గీకరించబడిన చిత్రాలు",

        "total_classes": "మొత్తం తరగతులు",

        "total_classes_text":
            "పంట వ్యాధి వర్గాలు",

        "per_class_accuracy": "ప్రతి తరగతి ఖచ్చితత్వం",

        "per_class_description":
            "ప్రతి పంట వ్యాధి వర్గానికి సాధించిన ఖచ్చితత్వం.",

        "corn_common_rust": "మొక్కజొన్న సాధారణ తుప్పు",
        "corn_northern_leaf_blight": "మొక్కజొన్న నార్తర్న్ లీఫ్ బ్లైట్",
        "corn_healthy": "ఆరోగ్యకరమైన మొక్కజొన్న",

        "potato_early_blight": "బంగాళాదుంప ఎర్లీ బ్లైట్",
        "potato_late_blight": "బంగాళాదుంప లేట్ బ్లైట్",
        "potato_healthy": "ఆరోగ్యకరమైన బంగాళాదుంప",

        "tomato_early_blight": "టమాటా ఎర్లీ బ్లైట్",
        "tomato_late_blight": "టమాటా లేట్ బ్లైట్",
        "tomato_healthy": "ఆరోగ్యకరమైన టమాటా",

        "confusion_matrix": "కన్ఫ్యూజన్ మ్యాట్రిక్స్",

        "confusion_description":
            "కన్ఫ్యూజన్ మ్యాట్రిక్స్ వాస్తవ తరగతులను అంచనా వేసిన తరగతులతో పోలుస్తుంది.",

        "actual_predicted": "వాస్తవ / అంచనా",

        "corn_rust": "మొక్కజొన్న తుప్పు",
        "corn_blight": "మొక్కజొన్న బ్లైట్",
        "corn_healthy": "ఆరోగ్యకరమైన మొక్కజొన్న",

        "potato_early": "బంగాళాదుంప ఎర్లీ",
        "potato_late": "బంగాళాదుంప లేట్",
        "potato_healthy": "ఆరోగ్యకరమైన బంగాళాదుంప",

        "tomato_early": "టమాటా ఎర్లీ",
        "tomato_late": "టమాటా లేట్",
        "tomato_healthy": "ఆరోగ్యకరమైన టమాటా",

        "try_detection": "🔍 వ్యాధి గుర్తింపును ప్రయత్నించండి",
        "back_home": "🏠 హోమ్‌కు తిరిగి వెళ్లండి",

        "about_tagline": "AGROZYEN AI గురించి",

        "about_title": "AI తో స్మార్ట్ వ్యవసాయం",

        "about_description":
            "AgroZyen AI అనేది AI ఆధారిత పంట వ్యాధి గుర్తింపు వ్యవస్థ. "
            "ఇది ఆకు చిత్రాలను ఉపయోగించి పంట వ్యాధులను త్వరగా గుర్తించడంలో సహాయపడుతుంది.",

        "about_project_title": "AgroZyen గురించి",

        "about_project_text":
            "AgroZyen AI కృత్రిమ మేధస్సును ఉపయోగించి పంట ఆకుల చిత్రాలను "
            "విశ్లేషిస్తుంది మరియు రైతులకు వ్యాధులను గుర్తించడంలో సహాయపడుతుంది.",

        "about_ai_title": "కృత్రిమ మేధస్సు",

        "about_ai_text":
            "మా AI మోడల్ పంట ఆకులలోని దృశ్య నమూనాలను విశ్లేషించి "
            "అత్యంత సంభావ్య వ్యాధి వర్గాన్ని అంచనా వేస్తుంది.",

        "about_farmer_title": "రైతుల కోసం రూపొందించబడింది",

        "about_farmer_text":
            "AgroZyen AI రైతులు పంట ఆరోగ్యాన్ని తనిఖీ చేయడానికి "
            "మరియు ఉపయోగకరమైన వ్యాధి సమాచారాన్ని పొందడానికి సులభమైన మార్గాన్ని అందిస్తుంది.",

        "about_process_tag": "AGROZYEN ఎలా పనిచేస్తుంది",

        "about_process_title": "సులభమైన AI ఆధారిత ప్రక్రియ",

        "about_process_description":
            "AgroZyen AI పంట ఆకులను విశ్లేషించి సంభావ్య వ్యాధులను "
            "గుర్తించడానికి మూడు సులభమైన దశలను ఉపయోగిస్తుంది.",

        "about_step1_title": "చిత్రాన్ని అప్‌లోడ్ చేయండి",

        "about_step1_text":
            "మీరు విశ్లేషించాలనుకుంటున్న పంట ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి.",

        "about_step2_title": "AI విశ్లేషణ",

        "about_step2_text":
            "శిక్షణ పొందిన AI మోడల్ అప్‌లోడ్ చేసిన ఆకు యొక్క దృశ్య లక్షణాలను విశ్లేషిస్తుంది.",

        "about_step3_title": "అంచనాను పొందండి",

        "about_step3_text":
            "సిస్టమ్ సంభావ్య వ్యాధి, నమ్మక స్థాయి మరియు ఉపయోగకరమైన సమాచారాన్ని అందిస్తుంది.",

        "about_technology_title": "AI సాంకేతికత",

        "about_technology_text":
            "AgroZyen AI పంట వ్యాధి నమూనాలను గుర్తించడానికి "
            "శిక్షణ పొందిన డీప్ లెర్నింగ్ ఇమేజ్ క్లాసిఫికేషన్ మోడల్‌ను ఉపయోగిస్తుంది.",

        "about_accuracy_title": "మోడల్ పనితీరు",

        "about_accuracy_text":
            "మోడల్ తొమ్మిది పంట ఆరోగ్యం మరియు వ్యాధి వర్గాల ధృవీకరణ డేటాతో మూల్యాంకనం చేయబడింది.",

        "about_cta_title": "మీ పంటను తనిఖీ చేయడానికి సిద్ధంగా ఉన్నారా?",

        "about_cta_text":
            "పంట ఆకు చిత్రాన్ని అప్‌లోడ్ చేసి AgroZyen AI దాన్ని విశ్లేషించనివ్వండి.",

        "detect_tagline": "AI వ్యాధి గుర్తింపు",

        "detect_title": "పంట వ్యాధిని గుర్తించండి",

        "detect_description":
            "పంట ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి. మా AI మోడల్ దానిని విశ్లేషించి "
            "సంభావ్య వ్యాధిని గుర్తిస్తుంది.",

        "select_image": "పంట ఆకు చిత్రాన్ని ఎంచుకోండి",

        "choose_image": "చిత్రాన్ని ఎంచుకోండి",

        "analyze_image": "🔍 చిత్రాన్ని విశ్లేషించండి",

        "supported_formats":
            "మద్దతు ఉన్న ఫార్మాట్లు: JPG, JPEG, PNG",

        "result_tagline": "AI గుర్తింపు ఫలితం",

        "result_title": "వ్యాధి గుర్తింపు ఫలితం",

        "prediction": "అంచనా",

        "disease_detected": "గుర్తించబడిన వ్యాధి",

        "healthy_crop": "ఆరోగ్యకరమైన పంట",

        "crop": "పంట",

        "confidence": "నమ్మకం",

        "confidence_level": "నమ్మక స్థాయి",

        "high": "అధికం",

        "moderate": "మధ్యస్థం",

        "low": "తక్కువ",

        "about_disease": "ఈ పరిస్థితి గురించి",

        "symptoms": "సాధారణ లక్షణాలు",

        "recommendation": "సిఫార్సు",

        "no_symptoms":
            "వ్యాధి లక్షణాలు గుర్తించబడలేదు.",

        "analyze_another":
            "🔍 మరొక చిత్రాన్ని విశ్లేషించండి",

        "return_home":
            "🏠 హోమ్‌కు తిరిగి వెళ్లండి",

        "prediction_error":
            "చిత్రాన్ని విశ్లేషించేటప్పుడు లోపం ఏర్పడింది.",

        "invalid_image":
            "దయచేసి మొక్కజొన్న, బంగాళాదుంప లేదా టమాటా ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి.",

        "invalid_image_title":
            "చెల్లని చిత్రం",

        "invalid_image_description":
            "అప్‌లోడ్ చేసిన చిత్రం సరైన పంట ఆకు చిత్రంగా కనిపించడం లేదు. "
            "దయచేసి మొక్కజొన్న, బంగాళాదుంప లేదా టమాటా ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి.",

        "footer":
            "AI ఆధారిత పంట వ్యాధి గుర్తింపు వ్యవస్థ."
    },


    # =====================================================
    # GUJARATI
    # =====================================================

    "gu": {

        "home": "હોમ",
        "detect": "રોગ શોધો",
        "performance": "પ્રદર્શન",
        "about": "અમારા વિશે",
        "get_started": "શરૂ કરો",

        "tagline": "🌿 AI આધારિત ખેતી",
        "hero_title": "તમારા પાકનું રક્ષણ કરો",
        "hero_ai": "AI સાથે",

        "description":
            "કૃત્રિમ બુદ્ધિનો ઉપયોગ કરીને પાકના રોગોને ઝડપથી શોધો. "
            "પાનની તસવીર અપલોડ કરો અને તરત જ રોગની આગાહી મેળવો.",

        "detect_button": "🔍 રોગ શોધો",
        "how_button": "તે કેવી રીતે કામ કરે છે →",

        "ai_detection": "AI રોગ શોધ",

        "ai_description":
            "પાકના પાનની તસવીર અપલોડ કરો અને AI તેનું વિશ્લેષણ કરવા દો.",

        "upload": "છબી અપલોડ કરો",

        "upload_text":
            "તમારા પાકના પાનની સ્પષ્ટ તસવીર અપલોડ કરો.",

        "analysis": "AI વિશ્લેષણ",

        "analysis_text":
            "અમારું AI મોડેલ પાનનું રોગ માટે વિશ્લેષણ કરે છે.",

        "results": "પરિણામ મેળવો",

        "results_text":
            "રોગની આગાહી અને વિશ્વાસ સ્તર મેળવો.",

        "simple_process": "સરળ પ્રક્રિયા",

        "how_it_works": "તે કેવી રીતે કામ કરે છે",

        "steps_description":
            "પાકના રોગોને શોધવા માટે માત્ર થોડા સરળ પગલાં જરૂરી છે.",

        "step_upload": "અપલોડ",

        "step_upload_text":
            "અસરગ્રસ્ત પાકના પાનનો ફોટો અપલોડ કરો.",

        "step_analyze": "વિશ્લેષણ",

        "step_analyze_text":
            "AI પાનના દૃશ્ય પેટર્નનું વિશ્લેષણ કરે છે.",

        "step_detect": "શોધો",

        "step_detect_text":
            "સિસ્ટમ સંભવિત રોગને ઓળખે છે.",

        "step_learn": "શીખો",

        "step_learn_text":
            "રોગની માહિતી અને નિવારણની ટીપ્સ જુઓ.",

        "performance_tagline": "AI મોડેલ મૂલ્યાંકન",

        "performance_title": "મોડેલનું પ્રદર્શન",

        "performance_description":
            "માન્યતા ડેટાસેટ પર AgroZyen AI રોગ વર્ગીકરણ મોડેલનું પ્રદર્શન.",

        "validation_accuracy": "માન્યતા ચોકસાઈ",

        "overall_accuracy": "મોડેલની કુલ ચોકસાઈ",

        "correct_predictions": "સાચી આગાહીઓ",

        "correct_predictions_text":
            "સચોટ રીતે વર્ગીકૃત કરેલી છબીઓ",

        "total_classes": "કુલ વર્ગો",

        "total_classes_text":
            "પાક રોગની શ્રેણીઓ",

        "per_class_accuracy": "દરેક વર્ગની ચોકસાઈ",

        "per_class_description":
            "દરેક પાક રોગ શ્રેણી માટે પ્રાપ્ત ચોકસાઈ.",

        "corn_common_rust": "મકાઈ સામાન્ય રસ્ટ",

        "corn_northern_leaf_blight":
            "મકાઈ નોર્ધર્ન લીફ બ્લાઇટ",

        "corn_healthy": "તંદુરસ્ત મકાઈ",

        "potato_early_blight": "બટાકા અર્લી બ્લાઇટ",

        "potato_late_blight": "બટાકા લેટ બ્લાઇટ",

        "potato_healthy": "તંદુરસ્ત બટાકા",

        "tomato_early_blight": "ટામેટા અર્લી બ્લાઇટ",

        "tomato_late_blight": "ટામેટા લેટ બ્લાઇટ",

        "tomato_healthy": "તંદુરસ્ત ટામેટા",

        "confusion_matrix": "કન્ફ્યુઝન મેટ્રિક્સ",

        "confusion_description":
            "કન્ફ્યુઝન મેટ્રિક્સ વાસ્તવિક વર્ગોની સરખામણી અનુમાનિત વર્ગો સાથે કરે છે.",

        "actual_predicted": "વાસ્તવિક / અનુમાનિત",

        "corn_rust": "મકાઈ રસ્ટ",

        "corn_blight": "મકાઈ બ્લાઇટ",

        "corn_healthy": "તંદુરસ્ત મકાઈ",

        "potato_early": "બટાકા અર્લી",

        "potato_late": "બટાકા લેટ",

        "potato_healthy": "તંદુરસ્ત બટાકા",

        "tomato_early": "ટામેટા અર્લી",

        "tomato_late": "ટામેટા લેટ",

        "tomato_healthy": "તંદુરસ્ત ટામેટા",

        "try_detection": "🔍 રોગ શોધવાનો પ્રયાસ કરો",

        "back_home": "🏠 હોમ પર પાછા જાઓ",

        "about_tagline": "AGROZYEN AI વિશે",

        "about_title": "AI સાથે સ્માર્ટ ખેતી",

        "about_description":
            "AgroZyen AI એ AI આધારિત પાક રોગ શોધ સિસ્ટમ છે "
            "જે પાનની છબીઓનો ઉપયોગ કરીને પાકના રોગોને ઝડપથી ઓળખવામાં મદદ કરે છે.",

        "about_project_title": "AgroZyen વિશે",

        "about_project_text":
            "AgroZyen AI કૃત્રિમ બુદ્ધિનો ઉપયોગ કરીને પાકના પાનની છબીઓનું "
            "વિશ્લેષણ કરે છે અને ખેડૂતોને રોગો ઓળખવામાં મદદ કરે છે.",

        "about_ai_title": "કૃત્રિમ બુદ્ધિ",

        "about_ai_text":
            "અમારું AI મોડેલ પાકના પાનમાં રહેલા દૃશ્ય પેટર્નનું "
            "વિશ્લેષણ કરીને સૌથી સંભવિત રોગની શ્રેણીનું અનુમાન કરે છે.",

        "about_farmer_title": "ખેડૂતો માટે બનાવેલ",

        "about_farmer_text":
            "AgroZyen AI ખેડૂતોને પાકના સ્વાસ્થ્યની તપાસ કરવા અને "
            "ઉપયોગી રોગ સંબંધિત માહિતી મેળવવાની સરળ રીત આપે છે.",

        "about_process_tag": "AGROZYEN કેવી રીતે કામ કરે છે",

        "about_process_title": "સરળ AI આધારિત પ્રક્રિયા",

        "about_process_description":
            "AgroZyen AI પાકના પાનનું વિશ્લેષણ કરવા અને સંભવિત રોગો "
            "ઓળખવા માટે ત્રણ સરળ પગલાંનો ઉપયોગ કરે છે.",

        "about_step1_title": "છબી અપલોડ કરો",

        "about_step1_text":
            "તમે જે પાકના પાનનું વિશ્લેષણ કરવા માંગો છો તેની સ્પષ્ટ છબી અપલોડ કરો.",

        "about_step2_title": "AI વિશ્લેષણ",

        "about_step2_text":
            "પ્રશિક્ષિત AI મોડેલ અપલોડ કરેલા પાનની દૃશ્ય વિશેષતાઓનું વિશ્લેષણ કરે છે.",

        "about_step3_title": "આગાહી મેળવો",

        "about_step3_text":
            "સિસ્ટમ સંભવિત રોગ, વિશ્વાસ સ્તર અને ઉપયોગી માહિતી પ્રદાન કરે છે.",

        "about_technology_title": "AI ટેકનોલોજી",

        "about_technology_text":
            "AgroZyen AI પાકના રોગના પેટર્નને ઓળખવા માટે "
            "પ્રશિક્ષિત ડીપ લર્નિંગ ઇમેજ ક્લાસિફિકેશન મોડેલનો ઉપયોગ કરે છે.",

        "about_accuracy_title": "મોડેલનું પ્રદર્શન",

        "about_accuracy_text":
            "મોડેલનું મૂલ્યાંકન નવ પાક આરોગ્ય અને રોગની શ્રેણીઓના "
            "માન્યતા ડેટાનો ઉપયોગ કરીને કરવામાં આવે છે.",

        "about_cta_title": "તમારો પાક તપાસવા માટે તૈયાર છો?",

        "about_cta_text":
            "પાકના પાનની છબી અપલોડ કરો અને AgroZyen AI તેનું વિશ્લેષણ કરવા દો.",

        "detect_tagline": "AI રોગ શોધ",

        "detect_title": "પાકનો રોગ શોધો",

        "detect_description":
            "પાકના પાનની છબી અપલોડ કરો અને અમારું AI મોડેલ "
            "તેનું વિશ્લેષણ કરીને સંભવિત રોગને ઓળખશે.",

        "select_image": "પાકના પાનની છબી પસંદ કરો",

        "choose_image": "છબી પસંદ કરો",

        "analyze_image": "🔍 છબીનું વિશ્લેષણ કરો",

        "supported_formats":
            "સપોર્ટેડ ફોર્મેટ: JPG, JPEG, PNG",

        "result_tagline": "AI શોધ પરિણામ",

        "result_title": "રોગ શોધ પરિણામ",

        "prediction": "આગાહી",

        "disease_detected": "શોધાયેલ રોગ",

        "healthy_crop": "તંદુરસ્ત પાક",

        "crop": "પાક",

        "confidence": "વિશ્વાસ",

        "confidence_level": "વિશ્વાસ સ્તર",

        "high": "ઉચ્ચ",

        "moderate": "મધ્યમ",

        "low": "ઓછું",

        "about_disease": "આ સ્થિતિ વિશે",

        "symptoms": "સામાન્ય લક્ષણો",

        "recommendation": "ભલામણ",

        "no_symptoms":
            "રોગના કોઈ લક્ષણો મળ્યા નથી.",

        "analyze_another":
            "🔍 બીજી છબીનું વિશ્લેષણ કરો",

        "return_home":
            "🏠 હોમ પર પાછા જાઓ",

        "prediction_error":
            "છબીનું વિશ્લેષણ કરતી વખતે ભૂલ થઈ.",

        "invalid_image":
            "કૃપા કરીને મકાઈ, બટાકા અથવા ટામેટાના પાનની સ્પષ્ટ તસવીર અપલોડ કરો.",

        "invalid_image_title":
            "અમાન્ય છબી",

        "invalid_image_description":
            "અપલોડ કરેલી છબી યોગ્ય પાકના પાનની છબી લાગતી નથી. "
            "કૃપા કરીને મકાઈ, બટાકા અથવા ટામેટાના પાનની સ્પષ્ટ તસવીર અપલોડ કરો.",

        "footer":
            "AI આધારિત પાક રોગ શોધ સિસ્ટમ."
    }
}


# =========================================================
# LANGUAGE CONTEXT
# =========================================================

@app.context_processor
def inject_language():

    language = session.get("language", "en")

    if language not in TRANSLATIONS:
        language = "en"

    return {
        "current_language": language,
        "t": TRANSLATIONS[language]
    }


# =========================================================
# CHANGE LANGUAGE
# =========================================================

@app.route("/set-language/<language>")
def set_language(language):

    supported_languages = [
        "en",
        "hi",
        "mr",
        "te",
        "gu"
    ]

    if language in supported_languages:
        session["language"] = language

    return redirect(request.referrer or "/")


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "model/crop_disease_model.onnx"

UPLOAD_FOLDER = "static/uploads"

REVIEWS_FILE = "reviews.json"

# Minimum confidence required.
# Increase to 80 or 85 if unwanted images are still getting results.
MIN_CONFIDENCE = 60.0


# =========================================================
# CREATE UPLOAD FOLDER
# =========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# LOAD REVIEWS
# =========================================================

def load_reviews():

    if not os.path.exists(REVIEWS_FILE):
        return []

    try:

        with open(
            REVIEWS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return []


# =========================================================
# SAVE REVIEWS
# =========================================================

def save_reviews(reviews):

    with open(
        REVIEWS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            reviews,
            f,
            ensure_ascii=False,
            indent=4
        )


# =========================================================
# LOAD ONNX MODEL
# =========================================================

try:

    model = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    MODEL_INPUT_NAME = model.get_inputs()[0].name

    print("Model loaded successfully.")

except Exception as e:

    model = None
    MODEL_INPUT_NAME = None

    print(
        "Model loading failed:",
        str(e)
    )

# =========================================================
# CLASS NAMES
# =========================================================

class_names = [

    "Corn___Common_rust",

    "Corn___Northern_Leaf_Blight",

    "Corn___healthy",

    "Potato___Early_blight",

    "Potato___Late_blight",

    "Potato___healthy",

    "Tomato___Early_blight",

    "Tomato___Late_blight",

    "Tomato___healthy"

]


# =========================================================
# DISEASE INFORMATION
# =========================================================

disease_info = {

    "Corn___Common_rust": {

        "about":
            "Common rust is a fungal disease that affects corn leaves.",

        "symptoms": [
            "Reddish-brown rust spots on leaves",
            "Yellowing around infected areas",
            "Reduced photosynthesis"
        ],

        "recommendation":
            "Use resistant varieties, maintain field sanitation, "
            "and monitor plants regularly."
    },


    "Corn___Northern_Leaf_Blight": {

        "about":
            "Northern Leaf Blight produces long gray-green or brown "
            "lesions on corn leaves.",

        "symptoms": [
            "Long cigar-shaped leaf lesions",
            "Brown or gray patches",
            "Premature leaf drying"
        ],

        "recommendation":
            "Use resistant varieties and remove infected crop debris."
    },


    "Corn___healthy": {

        "about":
            "The uploaded corn leaf appears healthy.",

        "symptoms": [],

        "recommendation":
            "Continue regular monitoring, proper irrigation, "
            "balanced nutrition, and good agricultural practices."
    },


    "Potato___Early_blight": {

        "about":
            "Potato Early Blight commonly affects potato leaves.",

        "symptoms": [
            "Dark circular spots on leaves",
            "Target-like rings inside lesions",
            "Yellowing of surrounding leaf tissue"
        ],

        "recommendation":
            "Remove infected leaves and improve air circulation."
    },


    "Potato___Late_blight": {

        "about":
            "Potato Late Blight can rapidly damage potato leaves, "
            "stems, and tubers.",

        "symptoms": [
            "Dark irregular leaf lesions",
            "Rapid browning of leaves",
            "White growth under humid conditions"
        ],

        "recommendation":
            "Monitor frequently and remove infected material."
    },


    "Potato___healthy": {

        "about":
            "The uploaded potato leaf appears healthy.",

        "symptoms": [],

        "recommendation":
            "Continue regular monitoring and maintain proper watering "
            "and nutrition."
    },


    "Tomato___Early_blight": {

        "about":
            "Tomato Early Blight commonly affects tomato leaves.",

        "symptoms": [
            "Dark spots on older leaves",
            "Concentric ring patterns",
            "Yellowing around affected areas",
            "Premature leaf drop"
        ],

        "recommendation":
            "Remove infected leaves and improve air circulation."
    },


    "Tomato___Late_blight": {

        "about":
            "Tomato Late Blight can rapidly damage tomato leaves, "
            "stems, and fruit.",

        "symptoms": [
            "Large dark irregular leaf patches",
            "Rapid browning",
            "Leaf death",
            "Dark lesions on fruit"
        ],

        "recommendation":
            "Remove severely infected material and improve air circulation."
    },


    "Tomato___healthy": {

        "about":
            "The uploaded tomato leaf appears healthy.",

        "symptoms": [],

        "recommendation":
            "Continue regular monitoring, proper watering, "
            "balanced nutrition, and good plant hygiene."
    }

}


# =========================================================
# DISEASE DISPLAY TRANSLATION
# =========================================================

DISEASE_TRANSLATION_KEYS = {

    "Corn___Common_rust":
        "corn_common_rust",

    "Corn___Northern_Leaf_Blight":
        "corn_northern_leaf_blight",

    "Corn___healthy":
        "corn_healthy",

    "Potato___Early_blight":
        "potato_early_blight",

    "Potato___Late_blight":
        "potato_late_blight",

    "Potato___healthy":
        "potato_healthy",

    "Tomato___Early_blight":
        "tomato_early_blight",

    "Tomato___Late_blight":
        "tomato_late_blight",

    "Tomato___healthy":
        "tomato_healthy"
}


# =========================================================
# TRANSLATED DISEASE INFORMATION
# =========================================================

# English information is used as fallback for now.
# Your UI translations still work for all 5 languages.

DISEASE_INFO_TRANSLATIONS = {

    "en": disease_info
}


# =========================================================
# HELPER — GET LANGUAGE
# =========================================================

def get_current_language():

    language = session.get(
        "language",
        "en"
    )

    if language not in TRANSLATIONS:
        language = "en"

    return language


# =========================================================
# HELPER — INVALID IMAGE PAGE
# =========================================================

def invalid_image_response():

    language = get_current_language()

    translations = TRANSLATIONS[language]

    return render_template(
        "result.html",

        image_path=None,

        disease=translations["invalid_image_title"],

        crop="",

        confidence=0,

        confidence_level=translations["low"],

        is_healthy=False,

        about=translations["invalid_image_description"],

        symptoms=[],

        recommendation=translations["invalid_image"],

        disease_key=None,

        confidence_key="low",

        invalid_image=True
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# DETECT
# =========================================================

@app.route("/detect")
def detect():

    return render_template(
        "detect.html"
    )


# =========================================================
# PERFORMANCE
# =========================================================

@app.route("/performance")
def performance():

    return render_template(
        "performance.html"
    )


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# SUPPORT
# =========================================================

@app.route(
    "/support",
    methods=["GET", "POST"]
)
def support():

    support_submitted = False

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        topic = request.form.get(
            "topic",
            ""
        ).strip()

        message = request.form.get(
            "message",
            ""
        ).strip()

        print(
            "\n========================================"
        )

        print(
            "NEW SUPPORT REQUEST"
        )

        print(
            "========================================"
        )

        print(
            "Name:",
            name
        )

        print(
            "Email:",
            email
        )

        print(
            "Topic:",
            topic
        )

        print(
            "Message:",
            message
        )

        print(
            "========================================\n"
        )

        support_submitted = True

    return render_template(
        "support.html",
        support_submitted=support_submitted
    )


# =========================================================
# PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------------------------------
    # CHECK MODEL
    # -----------------------------------------------------

    if model is None:

        return "AI model could not be loaded."


    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if "leafImage" not in request.files:

        return "No image uploaded."


    file = request.files["leafImage"]


    if file.filename == "":

        return "No image selected."


    # -----------------------------------------------------
    # CHECK FILE EXTENSION
    # -----------------------------------------------------

    allowed_extensions = {
        "jpg",
        "jpeg",
        "png"
    }

    original_filename = file.filename.lower()

    extension = original_filename.rsplit(
        ".",
        1
    )[-1]

    if extension not in allowed_extensions:

        return "Invalid image format. Please upload JPG, JPEG or PNG."


    # -----------------------------------------------------
    # SAVE FILE
    # -----------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    # Prevent empty filename
    if not filename:

        filename = "uploaded_leaf.jpg"


    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    try:

        file.save(
            image_path
        )


        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # -------------------------------------------------
        # BASIC IMAGE SIZE CHECK
        # -------------------------------------------------

        width, height = image.size

        if width < 100 or height < 100:

            return invalid_image_response()


        # -------------------------------------------------
        # RESIZE IMAGE
        # -------------------------------------------------

        image = image.resize(
            (224, 224)
        )


        # -------------------------------------------------
        # CONVERT TO NUMPY
        # -------------------------------------------------

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # -------------------------------------------------
        # NORMALIZATION
        # -------------------------------------------------
        #
        # IMPORTANT:
        # Your training code must use the same preprocessing.
        #
        # If your train_model.py uses:
        # Rescaling(1./255)
        # then keep this normalization.
        #

        image_array = image_array / 255.0


        # -------------------------------------------------
        # ADD BATCH DIMENSION
        # -------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        predictions = model.run(
    None,
    {
        MODEL_INPUT_NAME: image_array.astype(np.float32)
    }
)[0]


        # -------------------------------------------------
        # CHECK PREDICTION SHAPE
        # -------------------------------------------------

        if predictions is None or len(predictions) == 0:

            return invalid_image_response()


        probabilities = predictions[0]


        if len(probabilities) != len(class_names):

            print(
                "Prediction classes:",
                len(probabilities)
            )

            print(
                "Expected classes:",
                len(class_names)
            )

            return "Model output does not match class names."


        # -------------------------------------------------
        # HIGHEST PROBABILITY
        # -------------------------------------------------

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )


        disease = class_names[
            predicted_index
        ]


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = float(
            probabilities[
                predicted_index
            ] * 100
        )


        print(
            "Prediction:",
            disease
        )

        print(
            "Confidence:",
            confidence
        )


        # =================================================
        # IMPORTANT INVALID IMAGE CHECK
        # =================================================
        #
        # If confidence is too low, do NOT show a disease.
        #

        if confidence < MIN_CONFIDENCE:

            print(
                "⚠️ Image rejected because confidence is too low."
            )

            return invalid_image_response()


        # -------------------------------------------------
        # HEALTHY CHECK
        # -------------------------------------------------

        is_healthy = disease.endswith(
            "___healthy"
        )


        # -------------------------------------------------
        # CROP
        # -------------------------------------------------

        if disease.startswith(
            "Corn"
        ):

            crop = "Corn"

        elif disease.startswith(
            "Potato"
        ):

            crop = "Potato"

        elif disease.startswith(
            "Tomato"
        ):

            crop = "Tomato"

        else:

            crop = "Unknown"


        # -------------------------------------------------
        # CONFIDENCE LEVEL
        # -------------------------------------------------

        if confidence >= 80:

            confidence_level = "High"

            confidence_key = "high"

        elif confidence >= 60:

            confidence_level = "Moderate"

            confidence_key = "moderate"

        else:

            confidence_level = "Low"

            confidence_key = "low"


        # -------------------------------------------------
        # CURRENT LANGUAGE
        # -------------------------------------------------

        language = get_current_language()

        translations = TRANSLATIONS[
            language
        ]


        # -------------------------------------------------
        # TRANSLATED DISEASE NAME
        # -------------------------------------------------

        disease_key = DISEASE_TRANSLATION_KEYS.get(
            disease
        )


        if disease_key:

            disease_display = translations.get(
                disease_key,
                disease.replace(
                    "___",
                    " - "
                ).replace(
                    "_",
                    " "
                )
            )

        else:

            disease_display = disease.replace(
                "___",
                " - "
            ).replace(
                "_",
                " "
            )


        # -------------------------------------------------
        # TRANSLATED CROP
        # -------------------------------------------------

        crop_translations = {

            "Corn": {

                "en": "Corn",
                "hi": "मक्का",
                "mr": "मका",
                "te": "మొక్కజొన్న",
                "gu": "મકાઈ"
            },

            "Potato": {

                "en": "Potato",
                "hi": "आलू",
                "mr": "बटाटा",
                "te": "బంగాళాదుంప",
                "gu": "બટાકા"
            },

            "Tomato": {

                "en": "Tomato",
                "hi": "टमाटर",
                "mr": "टोमॅटो",
                "te": "టమాటా",
                "gu": "ટામેટા"
            }
        }


        crop_display = crop_translations.get(
            crop,
            {}
        ).get(
            language,
            crop
        )


        # -------------------------------------------------
        # DISEASE INFORMATION
        # -------------------------------------------------

        info = DISEASE_INFO_TRANSLATIONS.get(
            language,
            DISEASE_INFO_TRANSLATIONS["en"]
        ).get(
            disease,
            disease_info.get(
                disease,
                {
                    "about":
                        "No additional information available.",

                    "symptoms": [],

                    "recommendation":
                        "Monitor the plant regularly."
                }
            )
        )


        # -------------------------------------------------
        # RESULT PAGE
        # -------------------------------------------------

        return render_template(

            "result.html",

            image_path="/" + image_path.replace(
                "\\",
                "/"
            ),

            disease=disease_display,

            crop=crop_display,

            confidence=round(
                confidence,
                2
            ),

            confidence_level=translations.get(
                confidence_key,
                confidence_level
            ),

            is_healthy=is_healthy,

            about=info["about"],

            symptoms=info["symptoms"],

            recommendation=info["recommendation"],

            disease_key=disease_key,

            confidence_key=confidence_key,

            invalid_image=False

        )


    except Exception as e:

        print(
            "Prediction Error:",
            e
        )

        language = get_current_language()

        return TRANSLATIONS[
            language
        ].get(
            "prediction_error",
            "Error while analyzing image."
        )


# =========================================================
# REVIEW
# =========================================================

@app.route(
    "/review",
    methods=["GET", "POST"]
)
def review():

    reviews = load_reviews()


    # -----------------------------------------------------
    # SUBMIT REVIEW
    # -----------------------------------------------------

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        rating = request.form.get(
            "rating",
            ""
        ).strip()

        message = request.form.get(
            "message",
            ""
        ).strip()


        if not name or not rating or not message:

            return render_template(
                "review.html",
                reviews=reviews,
                submitted=False
            )


        new_review = {

            "name": name,

            "rating": rating,

            "message": message
        }


        reviews.append(
            new_review
        )


        save_reviews(
            reviews
        )


        return render_template(

            "review.html",

            reviews=reviews,

            submitted=True,

            name=name,

            rating=rating,

            message=message
        )


    # -----------------------------------------------------
    # REVIEW PAGE
    # -----------------------------------------------------

    return render_template(

        "review.html",

        reviews=reviews,

        submitted=False
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

