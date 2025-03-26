[app]

# Application properties
title = Smart BestFriend
package.name = smartbestfriend
package.domain = com.yourdomain
source.dir = .
version = 1.0.0

# Android specific
requirements = 
    python3,
    flet,
    gtts,
    speechrecognition,
    langchain-groq,
    nltk,
    numpy,
    scipy,
    scikit-learn,
    joblib,
    pytz,
    pyjnius,
    android

android.permissions = 
    INTERNET,
    RECORD_AUDIO,
    READ_EXTERNAL_STORAGE,
    WRITE_EXTERNAL_STORAGE

android.api = 34
android.ndk = 25b
android.sdk = 34
android.arch = arm64-v8a

# Add your pickle files
source.include_exts = py,png,jpg,kv,atlas,pkl

# Include ML models
include_pkl = logistic_regression.pkl,tfidfvectorizer.pkl,label_encoder.pkl