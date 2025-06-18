# 🚀 Blueprint-Managed SHAP Demographic Analytics v2.0

## 🎯 **Overview**
Creates a **separate blueprint-managed service** with your new demographic data pipeline, maintaining consistency with your existing blueprint-managed service.

## 📋 **Blueprint Deployment Steps**

### **Step 1: Create New GitHub Repository**
1. Go to: https://github.com/new
2. **Repository name**: `shap-demographic-analytics-v2`
3. **Description**: `Blueprint-managed SHAP microservice with ArcGIS demographic data - 3,983 zip codes, 546 features`
4. **Visibility**: Private (recommended)
5. **Don't initialize** with README, .gitignore, or license
6. Click **"Create repository"**

### **Step 2: Update Git Remote**
```bash
# Remove current remote
git remote remove origin

# Add new remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/shap-demographic-analytics-v2.git

# Verify new remote
git remote -v
```

### **Step 3: Push Blueprint Configuration**
```bash
# Add blueprint files
git add render-v2.yaml BLUEPRINT_DEPLOYMENT_GUIDE.md

# Commit the blueprint files
git commit -m "🚀 Blueprint-managed SHAP Demographic Analytics v2.0 - Complete ArcGIS pipeline"

# Push to new repository
git push -u origin main
```

## 🎯 **Render Blueprint Deployment**

### **Method 1: Auto-Deploy from Blueprint (Recommended)**
1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Connect Repository**: Link your new `shap-demographic-analytics-v2` repository
3. **Render will automatically detect** the `render-v2.yaml` blueprint
4. **Services will be created** based on the blueprint configuration:
   - **Web Service**: `shap-demographic-analytics-v2`
   - **Worker Service**: `shap-demographic-worker-v2` (optional)

### **Method 2: Manual Blueprint Import**
1. **Go to Render Dashboard**
2. **New Blueprint**: Click "New +" → "Blueprint"
3. **Connect Repository**: Select your new repository
4. **Blueprint File**: Render will detect `render-v2.yaml`
5. **Deploy**: Click "Apply" to deploy the blueprint

## 🔗 **Expected Service URLs**
```
Web Service: https://shap-demographic-analytics-v2.onrender.com
Worker Service: https://shap-demographic-worker-v2.onrender.com
```

## ✅ **Benefits of Blueprint Management**
- ✅ **Consistent deployment** - Same approach as existing service
- ✅ **Version control** - All configuration in git
- ✅ **Infrastructure as Code** - Reproducible deployments
- ✅ **Separate services** - No interference with existing service
- ✅ **Easy updates** - Push to git to update services

## 🎯 **Service Configuration**
Your blueprint-managed service includes:

### **Web Service Features**
- **Service Name**: `shap-demographic-analytics-v2`
- **Environment**: Python 3
- **Plan**: Starter (upgradeable)
- **Data**: 3,983 zip codes, 546 features
- **SHAP**: Pre-calculated values for instant analysis

### **Environment Variables**
- `SERVICE_NAME`: "SHAP Demographic Analytics v2.0"
- `SERVICE_VERSION`: "2.0.0"
- `DATA_SOURCE`: "ArcGIS_Synapse54_Vetements_56_Layers"
- `RECORD_COUNT`: "3983"
- `FEATURE_COUNT`: "546"
- `SKIP_MODEL_TRAINING`: "true"
- `USE_PRECALCULATED_SHAP`: "true"

## 📊 **Post-Deployment Verification**
```bash
# Test health endpoint
curl https://shap-demographic-analytics-v2.onrender.com/health

# Test SHAP analysis
curl https://shap-demographic-analytics-v2.onrender.com/shap?zip_code=10001

# Check service info
curl https://shap-demographic-analytics-v2.onrender.com/info
```

## 🔄 **Next Steps**
1. **Create new GitHub repository**
2. **Update git remote and push**
3. **Let Render auto-deploy from blueprint**
4. **Test the new service endpoints**
5. **Update your Project Configuration Manager** with new service URL
6. **Keep both services running** until migration is complete

## 🎉 **Ready for Production**
Your new blueprint-managed service will provide:
- ✅ **3,983 zip codes** of demographic analysis
- ✅ **546 features** from 56 ArcGIS layers
- ✅ **Athletic brand insights** (Nike, Adidas, Jordan, etc.)
- ✅ **Consumer behavior data** (shopping, sports activities)
- ✅ **Pre-calculated SHAP values** for instant analysis
- ✅ **Blueprint management** for consistent deployments 