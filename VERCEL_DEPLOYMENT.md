# Vercel Deployment Guide

## 🚨 **Issues Fixed**

### **1. Missing package.json**
- **Problem**: Vercel expected a `package.json` file for Node.js builds
- **Solution**: Created minimal `package.json` for static file serving

### **2. Incorrect API Configuration**
- **Problem**: `vercel.json` was pointing to `api/index.py` (FastAPI app) instead of serverless function
- **Solution**: Updated to use `api/analyze.py` (serverless function)

### **3. FastAPI vs Serverless Mismatch**
- **Problem**: FastAPI apps don't work directly in Vercel serverless functions
- **Solution**: Used the existing `api/analyze.py` serverless handler

### **4. Heavy Dependencies**
- **Problem**: Original `Backtester_1.py` had many dependencies that could timeout
- **Solution**: Created `Backtester_vercel.py` optimized for serverless environment

### **5. File Size Issues**
- **Problem**: Large data directories were being deployed
- **Solution**: Added `.vercelignore` to exclude unnecessary files

## 🔧 **Files Modified/Created**

### **New Files:**
- `package.json` - Minimal Node.js package file
- `Backtester_vercel.py` - Simplified backtester for Vercel
- `.vercelignore` - Excludes large data directories
- `test_vercel.py` - Test script to verify setup
- `VERCEL_DEPLOYMENT.md` - This guide

### **Modified Files:**
- `vercel.json` - Updated API routing and configuration
- `api/analyze.py` - Added NaN cleaning and improved error handling

## 🚀 **Deployment Steps**

### **1. Install Vercel CLI**
```bash
npm install -g vercel
```

### **2. Login to Vercel**
```bash
vercel login
```

### **3. Deploy**
```bash
# From the project root directory
vercel

# Follow the prompts:
# - Link to existing project? No
# - Project name: trading-analyzer (or your preferred name)
# - Directory: ./
# - Override settings? No
```

### **4. Production Deploy**
```bash
vercel --prod
```

## 📋 **Configuration Details**

### **vercel.json**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/analyze.py",
      "use": "@vercel/python",
      "config": {
        "pythonVersion": "3.11"
      }
    },
    {
      "src": "static/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/",
      "dest": "/static/index.html"
    },
    {
      "src": "/api/analyze",
      "dest": "/api/analyze.py"
    },
    {
      "src": "/(.*)",
      "dest": "/static/$1"
    }
  ],
  "functions": {
    "api/analyze.py": {
      "maxDuration": 30
    }
  }
}
```

### **Key Changes:**
- **API Route**: `/api/analyze` → `api/analyze.py`
- **Function Timeout**: 30 seconds (increased from default 10s)
- **Python Version**: 3.11
- **Static Files**: Served from `static/` directory

## 🧪 **Testing**

### **Local Test**
```bash
python test_vercel.py
```

### **Expected Output**
```
🧪 Testing Vercel serverless function...
📊 CSV size: 323 characters
✅ API Response Status: 200
📈 Strategy: SuperSignal_v7_RTH15_v43
💰 Net Profit: $3501.82
📊 Total Trades: 191
🎉 Vercel setup test successful!
```

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **Function Timeout**
   - **Cause**: Large CSV files or complex analysis
   - **Solution**: Increase `maxDuration` in `vercel.json`

2. **Import Errors**
   - **Cause**: Missing dependencies
   - **Solution**: Check `requirements.txt` includes all needed packages

3. **File Not Found**
   - **Cause**: Incorrect file paths in serverless environment
   - **Solution**: Use absolute paths with `/var/task` for Vercel

4. **Memory Issues**
   - **Cause**: Large data processing
   - **Solution**: Use `Backtester_vercel.py` (simplified version)

### **Debug Commands:**
```bash
# Check deployment logs
vercel logs

# Test locally
vercel dev

# Check function status
vercel functions list
```

## 📊 **Performance Optimizations**

### **Backtester_vercel.py Features:**
- ✅ **Non-interactive matplotlib backend** (`Agg`)
- ✅ **Simplified analytics** (reduced complexity)
- ✅ **Optimized file I/O** (uses `/tmp` directory)
- ✅ **Reduced memory footprint**
- ✅ **Faster execution** (removed heavy computations)

### **File Size Reductions:**
- ✅ **Excluded Backtests/** (959 files)
- ✅ **Excluded StrategyReports/** (large CSV files)
- ✅ **Excluded development files**
- ✅ **Excluded other platform configs**

## 🎯 **Expected Results**

After successful deployment:
- ✅ **Web Interface**: Accessible at `https://your-project.vercel.app`
- ✅ **API Endpoint**: `https://your-project.vercel.app/api/analyze`
- ✅ **File Upload**: Drag-and-drop CSV functionality
- ✅ **Analysis Results**: Charts, metrics, and reports
- ✅ **PDF Export**: Downloadable analysis reports

## 💰 **Cost Considerations**

- **Free Tier**: 100GB bandwidth, 100GB-hours serverless functions
- **Function Timeout**: 30 seconds (suitable for most analyses)
- **File Size Limit**: 50MB per request (sufficient for CSV files)
- **Perfect for**: Personal use, small teams, moderate traffic

## 🔄 **Updates**

To update your deployed app:
```bash
# Make changes to your code
# Then redeploy
vercel --prod
```

## 🎉 **Success!**

Your trading strategy analyzer should now be live on Vercel with:
- Professional web interface
- Comprehensive analysis capabilities
- Mobile-friendly design
- PDF report generation
- No server maintenance required
