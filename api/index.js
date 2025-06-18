const { spawn } = require('child_process');
const path = require('path');

export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  try {
    // Path to our Python script
    const pythonScript = path.join(process.cwd(), 'api', 'python_handler.py');
    
    // Prepare the request data for Python
    const requestData = {
      method: req.method,
      url: req.url,
      headers: req.headers,
      body: req.body || {}
    };

    // Spawn Python process
    const python = spawn('python3', [pythonScript], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    // Send request data to Python
    python.stdin.write(JSON.stringify(requestData));
    python.stdin.end();

    let output = '';
    let errorOutput = '';

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        console.error('Python process error:', errorOutput);
        res.status(500).json({ error: 'Internal server error', details: errorOutput });
        return;
      }

      try {
        const response = JSON.parse(output);
        res.status(response.statusCode || 200);
        
        // Set response headers
        if (response.headers) {
          Object.entries(response.headers).forEach(([key, value]) => {
            res.setHeader(key, value);
          });
        }

        res.json(response.body || response);
      } catch (parseError) {
        console.error('Failed to parse Python response:', parseError);
        res.status(500).json({ error: 'Failed to parse response' });
      }
    });

  } catch (error) {
    console.error('Handler error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}
