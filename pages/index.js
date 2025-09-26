import { useState } from 'react';
import Head from 'next/head';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export default function Home() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Use default values from Backtester_1.py
  const config = {
    timeframe: '180d:15m',
    capital: 2500.0,
    commission: 4.04
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
    setResults(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    
    try {
      // Read the CSV file
      const csvText = await file.text();
      
              const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          timeframe: config.timeframe,
          capital: config.capital,
          commission: config.commission,
          csvData: csvText
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        setResults(data);
      } else {
        setError(data.error || 'An error occurred during analysis');
      }
    } catch (error) {
      setError('Network error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!results) return;
    
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `trading-analysis-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadPDF = async () => {
    if (!results) return;

    try {
      // Create a new PDF document
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      let yPosition = 20;

      // Add title
      pdf.setFontSize(20);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Trading Strategy Analysis Report', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 15;

      // Add strategy info
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Strategy: ${results.strategy_name}`, 20, yPosition);
      yPosition += 8;
      pdf.text(`Total Trades: ${results.trades_count} | RTH Trades: ${results.trades_rth_count}`, 20, yPosition);
      yPosition += 15;

      // Add table of contents
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Table of Contents', 20, yPosition);
      yPosition += 10;

      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      const tocItems = [
        'Key Performance Metrics',
        'Detailed Metrics',
        'Visualizations',
        'Monthly Performance',
        'Session Analysis',
        'Day of Week Analysis',
        'Hold Time Analysis',
        'Top Trades Analysis',
        'Streak Analysis',
        'Executive Summary'
      ];

      tocItems.forEach((item, index) => {
        pdf.text(`${index + 1}. ${item}`, 25, yPosition);
        yPosition += 6;
      });
      yPosition += 10;

      // Add key metrics with better spacing
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Key Performance Metrics', 20, yPosition);
      yPosition += 12;

      pdf.setFontSize(11);
      pdf.setFont('helvetica', 'normal');
      const metrics = [
        `Net Profit: $${results.metrics.net_profit?.toFixed(2) || 'N/A'}`,
        `Total Return: ${results.metrics.total_return_pct?.toFixed(2) || 'N/A'}%`,
        `Win Rate: ${results.metrics.win_rate_pct?.toFixed(2) || 'N/A'}%`,
        `Profit Factor: ${results.metrics.profit_factor?.toFixed(2) || 'N/A'}`,
        `Max Drawdown: $${results.metrics.max_drawdown_dollars?.toFixed(2) || 'N/A'}`,
        `Sharpe Ratio: ${results.metrics.sharpe_annualized?.toFixed(2) || 'N/A'}`
      ];

      metrics.forEach(metric => {
        pdf.text(metric, 20, yPosition);
        yPosition += 8;
      });

      yPosition += 15;

      // Add detailed metrics
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Detailed Metrics', 20, yPosition);
      yPosition += 12;

      pdf.setFontSize(11);
      pdf.setFont('helvetica', 'normal');
      const detailedMetrics = [
        `Average Win: $${results.metrics.avg_win_dollars?.toFixed(2) || 'N/A'}`,
        `Average Loss: $${results.metrics.avg_loss_dollars?.toFixed(2) || 'N/A'}`,
        `Largest Win: $${results.metrics.largest_winning_trade?.toFixed(2) || 'N/A'}`,
        `Largest Loss: $${results.metrics.largest_losing_trade?.toFixed(2) || 'N/A'}`,
        `Expectancy per Trade: $${results.metrics.expectancy_per_trade_dollars?.toFixed(2) || 'N/A'}`,
        `Recovery Factor: ${results.metrics.recovery_factor?.toFixed(2) || 'N/A'}`
      ];

      detailedMetrics.forEach(metric => {
        pdf.text(metric, 20, yPosition);
        yPosition += 8;
      });

      yPosition += 15;

      // Add all charts
      const charts = [
        { key: 'equity_curve', title: 'Equity Curve' },
        { key: 'drawdown_curve', title: 'Drawdown Curve' },
        { key: 'pl_histogram', title: 'P&L Distribution' },
        { key: 'heatmap', title: 'Day of Week & Hour Heatmap' }
      ];

      charts.forEach(chart => {
        if (results.charts?.[chart.key]) {
          yPosition += 15;
          pdf.setFontSize(14);
          pdf.setFont('helvetica', 'bold');
          pdf.text(chart.title, 20, yPosition);
          yPosition += 12;

          // Convert base64 image to PDF with better sizing
          const imgData = results.charts[chart.key];
          const imgWidth = 170; // Slightly larger
          const imgHeight = 110; // Slightly larger
          
          // Check if we need a new page with more margin
          if (yPosition + imgHeight + 20 > pageHeight - 20) {
            pdf.addPage();
            yPosition = 20;
            // Redraw title on new page
            pdf.setFontSize(14);
            pdf.setFont('helvetica', 'bold');
            pdf.text(chart.title, 20, yPosition);
            yPosition += 12;
          }
          
          // Center the image on the page
          const centerX = (pageWidth - imgWidth) / 2;
          pdf.addImage(imgData, 'PNG', centerX, yPosition, imgWidth, imgHeight);
          yPosition += imgHeight + 20; // More space after chart
        }
      });

      // Helper function to check and add page breaks
      const checkPageBreak = (requiredSpace = 20) => {
        if (yPosition + requiredSpace > pageHeight - 20) {
          pdf.addPage();
          yPosition = 20;
          return true;
        }
        return false;
      };

      // Helper function to add data tables with better formatting
      const addDataTable = (title, data, maxRows = 10) => {
        if (!data || data.length === 0) return;
        
        // Check if we need a new page (more space for table)
        checkPageBreak(120);

        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text(title, 20, yPosition);
        yPosition += 15;

        pdf.setFontSize(7);
        pdf.setFont('helvetica', 'normal');
        
        // Add table headers
        const headers = Object.keys(data[0] || {});
        const availableWidth = pageWidth - 40;
        const colWidth = availableWidth / headers.length; // Equal column width distribution
        let xPosition = 20;
        
        // Draw header background
        pdf.setFillColor(240, 240, 240);
        pdf.rect(20, yPosition - 3, availableWidth, 10, 'F');
        
        headers.forEach((header, index) => {
          const text = header.length > 22 ? header.substring(0, 22) + '...' : header;
          pdf.text(text, xPosition + 1, yPosition + 3);
          xPosition += colWidth;
        });
        yPosition += 12;

        // Add table data with alternating row colors
        data.slice(0, maxRows).forEach((row, rowIndex) => {
          // Check if we need a new page for this row (more conservative)
          if (checkPageBreak(15)) {
            // Redraw headers on new page
            pdf.setFillColor(240, 240, 240);
            pdf.rect(20, yPosition - 3, availableWidth, 10, 'F');
            xPosition = 20;
            headers.forEach(header => {
              const text = header.length > 22 ? header.substring(0, 22) + '...' : header;
              pdf.text(text, xPosition + 1, yPosition + 3);
              xPosition += colWidth;
            });
            yPosition += 12;
          }
          
          // Alternate row colors
          if (rowIndex % 2 === 0) {
            pdf.setFillColor(250, 250, 250);
            pdf.rect(20, yPosition - 2, availableWidth, 8, 'F');
          }
          
          xPosition = 20;
          Object.values(row).forEach((value, colIndex) => {
            const text = String(value);
            // Better text truncation based on column width
            const maxChars = Math.floor(colWidth / 1.2); // More characters per column
            const displayText = text.length > maxChars ? text.substring(0, maxChars) + '...' : text;
            pdf.text(displayText, xPosition + 1, yPosition + 3);
            xPosition += colWidth;
          });
          yPosition += 10;
        });
        
        yPosition += 25; // More space after table
      };

      // Helper function for trade analysis tables with better formatting
      const addTradeTable = (title, data, maxRows = 5) => {
        if (!data || data.length === 0) {
          // Add a note if no data
          checkPageBreak(50);
          pdf.setFontSize(14);
          pdf.setFont('helvetica', 'bold');
          pdf.text(title, 20, yPosition);
          yPosition += 15;
          pdf.setFontSize(10);
          pdf.setFont('helvetica', 'normal');
          pdf.text('No data available', 20, yPosition);
          yPosition += 20;
          return;
        }
        
        checkPageBreak(100);

        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text(title, 20, yPosition);
        yPosition += 15;

        pdf.setFontSize(7);
        pdf.setFont('helvetica', 'normal');
        
        const headers = Object.keys(data[0] || {});
        const availableWidth = pageWidth - 40;
        const colWidth = availableWidth / headers.length;
        let xPosition = 20;
        
        // Draw header background
        pdf.setFillColor(240, 240, 240);
        pdf.rect(20, yPosition - 3, availableWidth, 10, 'F');
        
        headers.forEach((header, index) => {
          const text = header.length > 20 ? header.substring(0, 20) + '...' : header;
          pdf.text(text, xPosition + 1, yPosition + 3);
          xPosition += colWidth;
        });
        yPosition += 12;

        // Debug: Add data count info
        pdf.setFontSize(6);
        pdf.text(`Showing ${Math.min(data.length, maxRows)} of ${data.length} rows`, 20, yPosition);
        yPosition += 8;

        data.slice(0, maxRows).forEach((row, rowIndex) => {
          if (checkPageBreak(12)) {
            pdf.setFillColor(240, 240, 240);
            pdf.rect(20, yPosition - 3, availableWidth, 10, 'F');
            xPosition = 20;
            headers.forEach(header => {
              const text = header.length > 20 ? header.substring(0, 20) + '...' : header;
              pdf.text(text, xPosition + 1, yPosition + 3);
              xPosition += colWidth;
            });
            yPosition += 12;
          }
          
          if (rowIndex % 2 === 0) {
            pdf.setFillColor(250, 250, 250);
            pdf.rect(20, yPosition - 2, availableWidth, 8, 'F');
          }
          
          xPosition = 20;
          Object.values(row).forEach((value, colIndex) => {
            const text = String(value);
            const maxChars = Math.floor(colWidth / 1.2);
            const displayText = text.length > maxChars ? text.substring(0, maxChars) + '...' : text;
            pdf.text(displayText, xPosition + 1, yPosition + 3);
            xPosition += colWidth;
          });
          yPosition += 10;
        });
        
        yPosition += 20;
      };

      // Add all data tables
      addDataTable('Monthly Performance', results.data?.monthly_performance, 8);
      addDataTable('Session Analysis', results.data?.session_kpis, 6);
      addDataTable('Day of Week Analysis', results.data?.dow_kpis, 7);
      addDataTable('Hold Time Analysis', results.data?.hold_kpis, 6);
      
      // Debug trade data
      console.log('Trade data debug:', {
        top_best: results.data?.top_best_trades?.length || 0,
        top_worst: results.data?.top_worst_trades?.length || 0,
        max_win: results.data?.max_win_streak?.length || 0,
        max_loss: results.data?.max_loss_streak?.length || 0
      });
      
      addTradeTable('Top Best Trades', results.data?.top_best_trades, 5);
      addTradeTable('Top Worst Trades', results.data?.top_worst_trades, 5);
      addTradeTable('Max Win Streak', results.data?.max_win_streak, 3);
      addTradeTable('Max Loss Streak', results.data?.max_loss_streak, 3);

      // Add analytics report if available
      if (results.analytics_md) {
        if (yPosition + 50 > pageHeight - 20) {
          pdf.addPage();
          yPosition = 20;
        }

        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Executive Summary', 20, yPosition);
        yPosition += 15;

        pdf.setFontSize(10);
        pdf.setFont('helvetica', 'normal');
        
        // Process analytics markdown and add to PDF with better page management
        const analyticsLines = results.analytics_md.split('\n');
        analyticsLines.forEach(line => {
          if (!line.trim()) return;
          
          // Handle headers
          if (line.startsWith('##')) {
            const text = line.replace(/^#+\s*/, '').trim();
            
            // Check if we need a new page for header
            checkPageBreak(50);
            
            yPosition += 10;
            pdf.setFontSize(12);
            pdf.setFont('helvetica', 'bold');
            // Add underline for headers
            pdf.text(text, 20, yPosition);
            pdf.line(20, yPosition + 1, 20 + pdf.getTextWidth(text), yPosition + 1);
            yPosition += 12;
            pdf.setFontSize(10);
            pdf.setFont('helvetica', 'normal');
            return;
          }
          
          // Handle sub-headers
          if (line.startsWith('###')) {
            const text = line.replace(/^#+\s*/, '').trim();
            
            // Check if we need a new page for sub-header
            checkPageBreak(40);
            
            yPosition += 8;
            pdf.setFontSize(11);
            pdf.setFont('helvetica', 'bold');
            pdf.text(text, 25, yPosition);
            yPosition += 10;
            pdf.setFontSize(10);
            pdf.setFont('helvetica', 'normal');
            return;
          }
          
          // Handle bullet points
          if (line.startsWith('- ')) {
            const text = line.replace(/^-\s*/, '').trim();
            const cleanText = text.replace(/\*\*(.*?)\*\*/g, '$1'); // Remove bold formatting
            const lines = pdf.splitTextToSize(cleanText, pageWidth - 60);
            
            lines.forEach((textLine, index) => {
              // Check if we need a new page for this line
              checkPageBreak(25);
              const bullet = index === 0 ? '• ' : '  ';
              pdf.text(bullet + textLine, 30, yPosition);
              yPosition += 7;
            });
            return;
          }
          
          // Handle horizontal rules
          if (line.trim() === '---' || line.trim() === '***') {
            // Check if we need a new page for horizontal rule
            checkPageBreak(30);
            yPosition += 8;
            pdf.line(20, yPosition, pageWidth - 20, yPosition);
            yPosition += 12;
            return;
          }
          
          // Handle regular text
          const cleanText = line.replace(/\*\*(.*?)\*\*/g, '$1'); // Remove bold formatting
          const lines = pdf.splitTextToSize(cleanText, pageWidth - 40);
          
          lines.forEach(textLine => {
            // Check if we need a new page for this line
            checkPageBreak(25);
            pdf.text(textLine, 20, yPosition);
            yPosition += 7;
          });
        });
      }

      // Add summary page
      pdf.addPage();
      yPosition = 20;
      
      pdf.setFontSize(16);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Key Highlights Summary', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 20;

      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      
      const highlights = [
        `Strategy: ${results.strategy_name}`,
        `Total Trades Analyzed: ${results.trades_count}`,
        `RTH Trades: ${results.trades_rth_count}`,
        `Net Profit: $${results.metrics.net_profit?.toFixed(2) || 'N/A'}`,
        `Total Return: ${results.metrics.total_return_pct?.toFixed(2) || 'N/A'}%`,
        `Win Rate: ${results.metrics.win_rate_pct?.toFixed(2) || 'N/A'}%`,
        `Profit Factor: ${results.metrics.profit_factor?.toFixed(2) || 'N/A'}`,
        `Max Drawdown: $${results.metrics.max_drawdown_dollars?.toFixed(2) || 'N/A'}`,
        `Sharpe Ratio: ${results.metrics.sharpe_annualized?.toFixed(2) || 'N/A'}`,
        `Average Win: $${results.metrics.avg_win_dollars?.toFixed(2) || 'N/A'}`,
        `Average Loss: $${results.metrics.avg_loss_dollars?.toFixed(2) || 'N/A'}`,
        `Expectancy per Trade: $${results.metrics.expectancy_per_trade_dollars?.toFixed(2) || 'N/A'}`
      ];

      highlights.forEach(highlight => {
        pdf.text(`• ${highlight}`, 20, yPosition);
        yPosition += 8;
      });

      // Add footer
      const totalPages = pdf.internal.getNumberOfPages();
      for (let i = 1; i <= totalPages; i++) {
        pdf.setPage(i);
        pdf.setFontSize(8);
        pdf.setFont('helvetica', 'normal');
        pdf.text(`Generated on ${new Date().toLocaleDateString()}`, 20, pageHeight - 10);
        pdf.text(`Page ${i} of ${totalPages}`, pageWidth - 30, pageHeight - 10, { align: 'right' });
      }

      // Download the PDF
      const fileName = `trading-analysis-${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(fileName);

    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Error generating PDF. Please try again.');
    }
  };

  return (
    <div className="container">
      <Head>
        <title>Trading Strategy Analyzer</title>
        <meta name="description" content="Analyze ThinkorSwim trading strategy CSV files" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="header">
        <h1>📈 Trading Strategy Analyzer</h1>
        <p>Upload your ThinkorSwim CSV file to get comprehensive trading analysis</p>
      </header>
      
      <main className="main">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="file-upload">
            <label htmlFor="file-input" className="file-label">
              <div className="file-input-wrapper">
                <input
                  id="file-input"
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  required
                />
                <div className="file-input-display">
                  {file ? (
                    <div className="file-selected">
                      <span className="file-icon">📄</span>
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">({(file.size / 1024).toFixed(1)} KB)</span>
                    </div>
                  ) : (
                    <div className="file-placeholder">
                      <span className="upload-icon">📁</span>
                      <span>Click to select CSV file or drag & drop</span>
                    </div>
                  )}
                </div>
              </div>
            </label>
          </div>


          <button 
            type="submit" 
            disabled={loading || !file}
            className="analyze-button"
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              '🚀 Analyze Strategy'
            )}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {results && (
          <div className="results">
            <div className="results-header">
              <h2>📊 Analysis Results</h2>
              <div className="download-buttons">
                <button onClick={downloadPDF} className="download-button pdf-button">
                  📄 Download PDF Report
                </button>
                <button onClick={downloadResults} className="download-button json-button">
                  💾 Download JSON Data
                </button>
              </div>
            </div>
            
            <div className="strategy-info">
              <h3>Strategy: {results.strategy_name}</h3>
              <p>Total Trades: {results.trades_count} | RTH Trades: {results.trades_rth_count}</p>
            </div>

            <div className="metrics-grid">
              <div className="metric-card primary">
                <div className="metric-label">Net Profit</div>
                <div className="metric-value">
                  ${results.metrics.net_profit?.toFixed(2) || 'N/A'}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-label">Total Return</div>
                <div className="metric-value">
                  {results.metrics.total_return_pct?.toFixed(2) || 'N/A'}%
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-label">Win Rate</div>
                <div className="metric-value">
                  {results.metrics.win_rate_pct?.toFixed(2) || 'N/A'}%
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-label">Profit Factor</div>
                <div className="metric-value">
                  {results.metrics.profit_factor?.toFixed(2) || 'N/A'}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-label">Max Drawdown</div>
                <div className="metric-value">
                  ${results.metrics.max_drawdown_dollars?.toFixed(2) || 'N/A'}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-label">Sharpe Ratio</div>
                <div className="metric-value">
                  {results.metrics.sharpe_annualized?.toFixed(2) || 'N/A'}
                </div>
              </div>
            </div>

            <div className="charts">
              {results.charts?.equity_curve && (
                <div className="chart-container">
                  <h3>📈 Equity Curve</h3>
                  <img 
                    src={results.charts.equity_curve}
                    alt="Equity Curve"
                    className="chart-image"
                  />
                </div>
              )}

              {results.charts?.drawdown_curve && (
                <div className="chart-container">
                  <h3>📉 Drawdown Curve</h3>
                  <img 
                    src={results.charts.drawdown_curve}
                    alt="Drawdown Curve"
                    className="chart-image"
                  />
                </div>
              )}

              {results.charts?.pl_histogram && (
                <div className="chart-container">
                  <h3>📊 P/L Distribution</h3>
                  <img 
                    src={results.charts.pl_histogram}
                    alt="P/L Histogram"
                    className="chart-image"
                  />
                </div>
              )}
            </div>

            <div className="detailed-metrics">
              <h3>📋 Detailed Metrics</h3>
              <div className="metrics-table">
                <div className="metric-row">
                  <span>Average Win:</span>
                  <span>${results.metrics.avg_win_dollars?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="metric-row">
                  <span>Average Loss:</span>
                  <span>${results.metrics.avg_loss_dollars?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="metric-row">
                  <span>Largest Win:</span>
                  <span>${results.metrics.largest_winning_trade?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="metric-row">
                  <span>Largest Loss:</span>
                  <span>${results.metrics.largest_losing_trade?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="metric-row">
                  <span>Expectancy per Trade:</span>
                  <span>${results.metrics.expectancy_per_trade_dollars?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="metric-row">
                  <span>Recovery Factor:</span>
                  <span>{results.metrics.recovery_factor?.toFixed(2) || 'N/A'}</span>
                </div>
              </div>
            </div>

            {/* Additional Charts */}
            {results.charts?.heatmap && (
              <div className="charts">
                <div className="chart-container">
                  <h3>🔥 Day of Week & Hour Heatmap</h3>
                  <img 
                    src={results.charts.heatmap}
                    alt="Heatmap"
                    className="chart-image"
                  />
                </div>
              </div>
            )}

            {/* Monthly Performance */}
            {results.data?.monthly_performance?.length > 0 && (
              <div className="data-section">
                <h3>📅 Monthly Performance</h3>
                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {Object.keys(results.data.monthly_performance[0] || {}).map(key => (
                          <th key={key}>{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.data.monthly_performance.map((row, index) => (
                        <tr key={index}>
                          {Object.values(row).map((value, i) => (
                            <td key={i}>{value}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Session Analysis */}
            {results.data?.session_kpis?.length > 0 && (
              <div className="data-section">
                <h3>⏰ Session Analysis</h3>
                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {Object.keys(results.data.session_kpis[0] || {}).map(key => (
                          <th key={key}>{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.data.session_kpis.map((row, index) => (
                        <tr key={index}>
                          {Object.values(row).map((value, i) => (
                            <td key={i}>{value}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Day of Week Analysis */}
            {results.data?.dow_kpis?.length > 0 && (
              <div className="data-section">
                <h3>📆 Day of Week Analysis</h3>
                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {Object.keys(results.data.dow_kpis[0] || {}).map(key => (
                          <th key={key}>{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.data.dow_kpis.map((row, index) => (
                        <tr key={index}>
                          {Object.values(row).map((value, i) => (
                            <td key={i}>{value}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Hold Time Analysis */}
            {results.data?.hold_kpis?.length > 0 && (
              <div className="data-section">
                <h3>⏱️ Hold Time Analysis</h3>
                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {Object.keys(results.data.hold_kpis[0] || {}).map(key => (
                          <th key={key}>{key}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.data.hold_kpis.map((row, index) => (
                        <tr key={index}>
                          {Object.values(row).map((value, i) => (
                            <td key={i}>{value}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Top Trades */}
            <div className="data-section">
              <h3>🏆 Top Trades Analysis</h3>
              <div className="trades-vertical">
                {results.data?.top_best_trades?.length > 0 && (
                  <div className="trades-container">
                    <h4>🥇 Best Trades</h4>
                    <div className="table-container">
                      <table className="data-table">
                        <thead>
                          <tr>
                            {Object.keys(results.data.top_best_trades[0] || {}).map(key => (
                              <th key={key}>{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {results.data.top_best_trades.slice(0, 5).map((row, index) => (
                            <tr key={index}>
                              {Object.values(row).map((value, i) => (
                                <td key={i}>{value}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                
                {results.data?.top_worst_trades?.length > 0 && (
                  <div className="trades-container">
                    <h4>🥉 Worst Trades</h4>
                    <div className="table-container">
                      <table className="data-table">
                        <thead>
                          <tr>
                            {Object.keys(results.data.top_worst_trades[0] || {}).map(key => (
                              <th key={key}>{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {results.data.top_worst_trades.slice(0, 5).map((row, index) => (
                            <tr key={index}>
                              {Object.values(row).map((value, i) => (
                                <td key={i}>{value}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Streak Analysis */}
            <div className="data-section">
              <h3>🔥 Streak Analysis</h3>
              <div className="streaks-vertical">
                {results.data?.max_win_streak?.length > 0 && (
                  <div className="streak-container">
                    <h4>📈 Max Win Streak</h4>
                    <div className="table-container">
                      <table className="data-table">
                        <thead>
                          <tr>
                            {Object.keys(results.data.max_win_streak[0] || {}).map(key => (
                              <th key={key}>{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {results.data.max_win_streak.slice(0, 3).map((row, index) => (
                            <tr key={index}>
                              {Object.values(row).map((value, i) => (
                                <td key={i}>{value}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                
                {results.data?.max_loss_streak?.length > 0 && (
                  <div className="streak-container">
                    <h4>📉 Max Loss Streak</h4>
                    <div className="table-container">
                      <table className="data-table">
                        <thead>
                          <tr>
                            {Object.keys(results.data.max_loss_streak[0] || {}).map(key => (
                              <th key={key}>{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {results.data.max_loss_streak.slice(0, 3).map((row, index) => (
                            <tr key={index}>
                              {Object.values(row).map((value, i) => (
                                <td key={i}>{value}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Analytics Report */}
            {results.analytics_md && (
              <div className="data-section">
                <h3>📋 Executive Summary</h3>
                <div className="analytics-content">
                  {results.analytics_md.split('\n').map((line, index) => {
                    // Skip empty lines
                    if (!line.trim()) return null;
                    
                    // Headers (##)
                    if (line.startsWith('##')) {
                      const text = line.replace(/^#+\s*/, '').trim();
                      return <h4 key={index} className="analytics-header">{text}</h4>;
                    }
                    
                    // Sub-headers (###)
                    if (line.startsWith('###')) {
                      const text = line.replace(/^#+\s*/, '').trim();
                      return <h5 key={index} className="analytics-subheader">{text}</h5>;
                    }
                    
                    // Bullet points (-)
                    if (line.startsWith('- ')) {
                      const text = line.replace(/^-\s*/, '').trim();
                      // Handle **bold** formatting
                      const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                      return <div key={index} className="analytics-bullet" dangerouslySetInnerHTML={{__html: `• ${formattedText}`}} />;
                    }
                    
                    // Numbered lists
                    if (line.match(/^\d+\.\s/)) {
                      const text = line.replace(/^\d+\.\s*/, '').trim();
                      const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                      return <div key={index} className="analytics-numbered" dangerouslySetInnerHTML={{__html: `${line.split('.')[0]}. ${formattedText}`}} />;
                    }
                    
                    // Horizontal rules (---)
                    if (line.trim() === '---' || line.trim() === '***') {
                      return <hr key={index} className="analytics-divider" />;
                    }
                    
                    // Regular paragraphs
                    const formattedText = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                    return <p key={index} className="analytics-paragraph" dangerouslySetInnerHTML={{__html: formattedText}} />;
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Built for ThinkorSwim strategy analysis • All calculations use RTH (09:30-16:00 ET)</p>
      </footer>
    </div>
  );
}

