# Trading Strategy Analyzer - Deployment Guide

## 🚀 Quick Deploy to Vercel

### Prerequisites
- Node.js installed on your machine
- Vercel account (free at [vercel.com](https://vercel.com))
- Your ThinkorSwim CSV files

### Step 1: Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Vercel CLI globally
npm install -g vercel
```

### Step 2: Deploy to Vercel

```bash
# Login to Vercel (first time only)
vercel login

# Deploy the project
vercel

# Follow the prompts:
# - Link to existing project? No
# - Project name: trading-analyzer (or your preferred name)
# - Directory: ./
# - Override settings? No
```

### Step 3: Access Your App

After deployment, Vercel will give you a URL like:
```
https://trading-analyzer-abc123.vercel.app
```

## 📁 Project Structure

```
DataAnalyzer/
├── pages/
│   ├── api/
│   │   └── analyze.py          # Python API endpoint
│   ├── index.js                # Main web interface
│   └── _app.js                 # Next.js app wrapper
├── styles/
│   └── globals.css             # Styling
├── vercel.json                 # Vercel configuration
├── package.json                # Node.js dependencies
├── requirements.txt            # Python dependencies
└── Backtester_1.py            # Your original backtester
```

## 🔧 How It Works

1. **Frontend**: Next.js web interface with drag-and-drop file upload
2. **Backend**: Python API endpoint that processes CSV files
3. **Analysis**: Uses your existing `Backtester_1.py` logic
4. **Results**: Displays metrics, charts, and downloadable results

## 💰 Cost

- **Free Tier**: $0/month
- **Limits**: 100GB bandwidth, 100GB-hours serverless functions
- **Perfect for**: Personal use, small teams

## 🎯 Features

### For Your Cofounder:
- ✅ Drag & drop CSV files
- ✅ No technical setup required
- ✅ Professional web interface
- ✅ Download results as JSON
- ✅ Mobile-friendly

### Analysis Features:
- ✅ All your existing metrics
- ✅ Equity curve visualization
- ✅ Drawdown analysis
- ✅ P/L distribution
- ✅ RTH-only analysis (09:30-16:00 ET)
- ✅ Stop-loss normalization

## 🔄 Updates

To update your deployed app:

```bash
# Make changes to your code
# Then redeploy
vercel --prod
```

## 🛠️ Customization

### Change Default Parameters
Edit `pages/index.js`:
```javascript
const [config, setConfig] = useState({
  timeframe: '180d:15m',    // Change default timeframe
  capital: 2500.0,          // Change default capital
  commission: 4.04          // Change default commission
});
```

### Add More Analysis Options
Edit `pages/api/analyze.py` to add more configuration options.

### Custom Styling
Edit `styles/globals.css` to change colors, fonts, layout.

## 🐛 Troubleshooting

### Common Issues:

1. **"Module not found" errors**
   ```bash
   npm install
   ```

2. **Python dependencies not found**
   - Check `requirements.txt` is in root directory
   - Ensure all imports are in the API file

3. **File upload not working**
   - Check file size (Vercel has limits)
   - Ensure CSV format matches ThinkorSwim export

4. **Charts not displaying**
   - Check browser console for errors
   - Ensure matplotlib is working in serverless environment

### Debug Mode:
```bash
# Run locally for testing
npm run dev
```

## 📞 Support

If you need help:
1. Check Vercel logs: `vercel logs`
2. Test locally first: `npm run dev`
3. Check browser console for errors

## 🎉 Success!

Once deployed, your cofounder can:
1. Go to your Vercel URL
2. Upload ThinkorSwim CSV files
3. Get instant analysis results
4. Download results for further analysis

No more GitHub downloads or command line needed! 🚀

