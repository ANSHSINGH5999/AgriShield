"""PlantDoc -> PlantVillage label mapping (written by hand from the class names; reviewed against both datasets).

status:
  exact      - same crop and same condition
  ambiguous  - plausible but not certain (excluded from the STRICT external evaluation, kept in EXTENDED)
  no_match   - PlantVillage has no such class (excluded)
"""
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": {"plantvillage": "Apple___Apple_scab", "status": "exact"},
    "Apple leaf": {"plantvillage": "Apple___healthy", "status": "exact"},
    "Apple rust leaf": {"plantvillage": "Apple___Cedar_apple_rust", "status": "exact",
                        "note": "PlantVillage's rust class is cedar apple rust, the common apple rust"},
    "Bell_pepper leaf": {"plantvillage": "Pepper,_bell___healthy", "status": "exact"},
    "Bell_pepper leaf spot": {"plantvillage": "Pepper,_bell___Bacterial_spot", "status": "ambiguous",
                              "note": "'leaf spot' does not name the pathogen; PlantVillage only has bacterial spot"},
    "Blueberry leaf": {"plantvillage": "Blueberry___healthy", "status": "exact"},
    "Cherry leaf": {"plantvillage": "Cherry_(including_sour)___healthy", "status": "exact"},
    "Corn Gray leaf spot": {"plantvillage": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "status": "exact"},
    "Corn leaf blight": {"plantvillage": "Corn_(maize)___Northern_Leaf_Blight", "status": "ambiguous",
                         "note": "PlantDoc does not say northern or southern leaf blight"},
    "Corn rust leaf": {"plantvillage": "Corn_(maize)___Common_rust_", "status": "exact"},
    "Peach leaf": {"plantvillage": "Peach___healthy", "status": "exact"},
    "Potato leaf": {"plantvillage": "Potato___healthy", "status": "exact"},
    "Potato leaf early blight": {"plantvillage": "Potato___Early_blight", "status": "exact"},
    "Potato leaf late blight": {"plantvillage": "Potato___Late_blight", "status": "exact"},
    "Raspberry leaf": {"plantvillage": "Raspberry___healthy", "status": "exact"},
    "Soyabean leaf": {"plantvillage": "Soybean___healthy", "status": "exact"},
    "Soybean leaf": {"plantvillage": "Soybean___healthy", "status": "exact"},
    "Squash Powdery mildew leaf": {"plantvillage": "Squash___Powdery_mildew", "status": "exact"},
    "Strawberry leaf": {"plantvillage": "Strawberry___healthy", "status": "exact"},
    "Tomato Early blight leaf": {"plantvillage": "Tomato___Early_blight", "status": "exact"},
    "Tomato Septoria leaf spot": {"plantvillage": "Tomato___Septoria_leaf_spot", "status": "exact"},
    "Tomato leaf": {"plantvillage": "Tomato___healthy", "status": "exact"},
    "Tomato leaf bacterial spot": {"plantvillage": "Tomato___Bacterial_spot", "status": "exact"},
    "Tomato leaf late blight": {"plantvillage": "Tomato___Late_blight", "status": "exact"},
    "Tomato leaf mosaic virus": {"plantvillage": "Tomato___Tomato_mosaic_virus", "status": "exact"},
    "Tomato leaf yellow virus": {"plantvillage": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "status": "ambiguous",
                                 "note": "'yellow virus' most likely means yellow leaf curl virus, but is not explicit"},
    "Tomato mold leaf": {"plantvillage": "Tomato___Leaf_Mold", "status": "exact"},
    "Tomato two spotted spider mites leaf": {"plantvillage": "Tomato___Spider_mites Two-spotted_spider_mite", "status": "exact"},
    "grape leaf": {"plantvillage": "Grape___healthy", "status": "exact"},
    "grape leaf black rot": {"plantvillage": "Grape___Black_rot", "status": "exact"},
}
