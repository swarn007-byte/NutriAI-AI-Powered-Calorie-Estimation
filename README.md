# 🍎 NutriAI: AI-Powered Calorie Estimation  

NutriAI is a deep learning–powered system that estimates **calories and nutrients directly from food images**.  
It combines computer vision and nutritional science to make dietary tracking more accurate, automated, and user-friendly.  

---

## ✨ Features  
- 🔍 **Food Recognition** – Uses **YOLO** for real-time object detection of multiple food items.  
- 📏 **Portion Size Estimation** – Depth estimation techniques to determine serving size and volume.  
- 🧠 **Fine-Grained Classification** – **ResNet-based models** improve accuracy across similar food categories.  
- 📊 **Nutrition Mapping** – Automatically maps predictions to nutritional databases for a **detailed breakdown of macro- and micronutrients**.  

---

## 🛠️ Tech Stack  
- **Frameworks**: PyTorch / TensorFlow  
- **Models**: YOLO (object detection), ResNet (classification)  
- **Computer Vision**: Depth estimation for portion size  
- **Databases**: Nutrition APIs & datasets (e.g., USDA, FoodData Central)  
- **Deployment**: Python, FastAPI / Flask for API integration  

---

## 🚀 How It Works  
1. **Input an Image** of a meal.  
2. **YOLO detects** food items on the plate.  
3. **Depth estimation** calculates portion sizes.  
4. **ResNet classifier** refines recognition of food types.  
5. **Nutrition database mapping** provides calorie and nutrient breakdown.  

---

## 📷 Example Workflow  

```mermaid
flowchart LR
    A[Food Image] --> B[YOLO: Food Detection]
    B --> C[Depth Estimation: Portion Size]
    C --> D[ResNet: Fine-grained Classification]
    D --> E[Nutrition Database Mapping]
    E --> F[Calorie & Nutrient Report]
