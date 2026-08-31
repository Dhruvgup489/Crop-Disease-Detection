// AgroZyen AI Language System

const translations = {

    English: {
        home: "Home",
        detect: "Detect Disease",
        performance: "Performance",
        about: "About",
        getStarted: "Get Started",
        title: "Protect Your Crops With AI",
        description:
            "Detect crop diseases quickly using artificial intelligence.",
        detectButton: "🔍 Detect Disease",
        howWorks: "How It Works →"
    },

    Hindi: {
        home: "होम",
        detect: "रोग पहचानें",
        performance: "प्रदर्शन",
        about: "हमारे बारे में",
        getStarted: "शुरू करें",
        title: "AI के साथ अपनी फसलों की सुरक्षा करें",
        description:
            "कृत्रिम बुद्धिमत्ता का उपयोग करके फसल रोगों की तुरंत पहचान करें।",
        detectButton: "🔍 रोग पहचानें",
        howWorks: "यह कैसे काम करता है →"
    },

    Telugu: {
        home: "హోమ్",
        detect: "వ్యాధిని గుర్తించండి",
        performance: "పనితీరు",
        about: "మా గురించి",
        getStarted: "ప్రారంభించండి",
        title: "AIతో మీ పంటలను రక్షించండి",
        description:
            "కృత్రిమ మేధస్సును ఉపయోగించి పంట వ్యాధులను త్వరగా గుర్తించండి.",
        detectButton: "🔍 వ్యాధిని గుర్తించండి",
        howWorks: "ఇది ఎలా పనిచేస్తుంది →"
    },

    Gujarati: {
        home: "હોમ",
        detect: "રોગ શોધો",
        performance: "પ્રદર્શન",
        about: "અમારા વિશે",
        getStarted: "શરૂ કરો",
        title: "AI સાથે તમારા પાકનું રક્ષણ કરો",
        description:
            "કૃત્રિમ બુદ્ધિનો ઉપયોગ કરીને પાકના રોગોને ઝડપથી શોધો.",
        detectButton: "🔍 રોગ શોધો",
        howWorks: "તે કેવી રીતે કામ કરે છે →"
    }

};


// Change language
function changeLanguage(language) {

    if (!translations[language]) {
        console.error("Language not found:", language);
        return;
    }

    const text = translations[language];

    // Navigation
    const home = document.querySelector("#nav-home");
    const detect = document.querySelector("#nav-detect");
    const performance = document.querySelector("#nav-performance");
    const about = document.querySelector("#nav-about");
    const getStarted = document.querySelector("#nav-start");

    if (home) home.textContent = text.home;
    if (detect) detect.textContent = text.detect;
    if (performance) performance.textContent = text.performance;
    if (about) about.textContent = text.about;
    if (getStarted) getStarted.textContent = text.getStarted;

    // Hero
    const title = document.querySelector("#hero-title");
    const description = document.querySelector("#hero-description");
    const detectButton = document.querySelector("#hero-detect");
    const howWorks = document.querySelector("#hero-how");

    if (title) title.textContent = text.title;
    if (description) description.textContent = text.description;
    if (detectButton) detectButton.textContent = text.detectButton;
    if (howWorks) howWorks.textContent = text.howWorks;

    // Save selected language
    localStorage.setItem("agrozyenLanguage", language);
}


// Load saved language
document.addEventListener("DOMContentLoaded", function () {

    const savedLanguage =
        localStorage.getItem("agrozyenLanguage") || "English";

    const selector =
        document.querySelector("#languageSelector");

    if (selector) {
        selector.value = savedLanguage;

        selector.addEventListener("change", function () {
            changeLanguage(this.value);
        });
    }

    changeLanguage(savedLanguage);

});