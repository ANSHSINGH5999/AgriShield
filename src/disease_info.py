"""Plain-English explanation of every class the model can predict (general information, not a diagnosis).

Keys are the exact class names from models/class_names.json. Each entry: (cause type, simple description).
"""
HEALTHY = ("Healthy", "No disease was detected. The leaf looks healthy.")

DISEASE_INFO = {
    "Apple___Apple_scab": ("Fungus", "Dark, olive-green to black, rough spots on the leaves. Badly affected leaves can turn yellow and fall early."),
    "Apple___Black_rot": ("Fungus", "Small purple spots that grow into brown patches with a darker edge on the leaves. It can also rot the fruit."),
    "Apple___Cedar_apple_rust": ("Fungus", "Bright yellow-orange spots on the leaves. The fungus spends part of its life on nearby cedar or juniper trees."),
    "Apple___healthy": HEALTHY,
    "Blueberry___healthy": HEALTHY,
    "Cherry_(including_sour)___Powdery_mildew": ("Fungus", "A white, powder-like coating on the leaves. Leaves may curl and grow poorly."),
    "Cherry_(including_sour)___healthy": HEALTHY,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Fungus", "Long, narrow, grey-brown rectangular spots running between the leaf veins."),
    "Corn_(maize)___Common_rust_": ("Fungus", "Small reddish-brown, powdery bumps (pustules) scattered on both sides of the leaf, like rust."),
    "Corn_(maize)___Northern_Leaf_Blight": ("Fungus", "Long, cigar-shaped grey-green to tan patches on the leaves."),
    "Corn_(maize)___healthy": HEALTHY,
    "Grape___Black_rot": ("Fungus", "Brown leaf spots with a dark border. The grapes can shrivel into hard, black, raisin-like berries."),
    "Grape___Esca_(Black_Measles)": ("Fungus", "Yellow or red-brown 'tiger stripe' patterns between the leaf veins, and small dark spots on the grapes."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Fungus", "Irregular dark brown spots on the leaves. Badly affected leaves dry out and fall."),
    "Grape___healthy": HEALTHY,
    "Orange___Haunglongbing_(Citrus_greening)": ("Bacteria", "Blotchy, uneven yellowing of the leaves. Fruit stays small, green and bitter. It is spread by a tiny insect and has no cure."),
    "Peach___Bacterial_spot": ("Bacteria", "Small dark, angular spots on the leaves. The spots can fall out and leave small holes."),
    "Peach___healthy": HEALTHY,
    "Pepper,_bell___Bacterial_spot": ("Bacteria", "Small, water-soaked spots that turn brown. Affected leaves may turn yellow and drop."),
    "Pepper,_bell___healthy": HEALTHY,
    "Potato___Early_blight": ("Fungus", "Brown spots with rings inside, like a target, usually starting on older, lower leaves."),
    "Potato___Late_blight": ("Water mould", "Large, dark, wet-looking patches that spread quickly in cool, damp weather. It can destroy a whole crop."),
    "Potato___healthy": HEALTHY,
    "Raspberry___healthy": HEALTHY,
    "Soybean___healthy": HEALTHY,
    "Squash___Powdery_mildew": ("Fungus", "A white, powder-like coating on the leaves, which can turn yellow and dry out."),
    "Strawberry___Leaf_scorch": ("Fungus", "Many small purple spots that join together until the leaf looks burnt or scorched."),
    "Strawberry___healthy": HEALTHY,
    "Tomato___Bacterial_spot": ("Bacteria", "Small, dark, greasy-looking spots on the leaves, sometimes with a yellow ring around them."),
    "Tomato___Early_blight": ("Fungus", "Brown spots with rings inside, like a target, starting on the older, lower leaves."),
    "Tomato___Late_blight": ("Water mould", "Large, dark, wet-looking patches that spread fast in cool, damp weather. A white fuzz can appear underneath."),
    "Tomato___Leaf_Mold": ("Fungus", "Yellow patches on top of the leaf and an olive-green, fuzzy mould underneath. It is common in humid conditions."),
    "Tomato___Septoria_leaf_spot": ("Fungus", "Many small round spots with grey centres and dark edges, starting on the lower leaves."),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Pest", "Not a disease. Tiny mites suck the leaf's juice, leaving fine yellow speckles and sometimes thin webbing."),
    "Tomato___Target_Spot": ("Fungus", "Brown spots with rings inside, like a target, that can join into larger dead areas."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Virus", "Leaves curl upward and turn yellow, and the plant stays small. It is spread by whiteflies."),
    "Tomato___Tomato_mosaic_virus": ("Virus", "A patchy light and dark green 'mosaic' pattern, with twisted leaves. It spreads easily by touch and on tools."),
    "Tomato___healthy": HEALTHY,
}


def describe(class_name: str) -> tuple[str, str]:
    """(cause type, plain-English description); a neutral fallback if a class is ever missing."""
    return DISEASE_INFO.get(class_name, ("Unknown", "No description available for this class."))
