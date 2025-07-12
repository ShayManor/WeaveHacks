const fetch = require('node-fetch');

exports.googleHomeWebhook = async (req, res) => {
  // Enable CORS for web requests
  res.set('Access-Control-Allow-Origin', '*');
  res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.status(204).send('');
    return;
  }

  try {
    console.log('Received request:', JSON.stringify(req.body, null, 2));
    
    // Handle different types of Google Smart Home requests
    if (req.body.inputs && req.body.inputs[0]) {
      const input = req.body.inputs[0];
      
      switch (input.intent) {
        case 'action.devices.SYNC':
          // Return device sync response
          return handleDeviceSync(req, res);
          
        case 'action.devices.QUERY':
          // Return device state
          return handleDeviceQuery(req, res);
          
        case 'action.devices.EXECUTE':
          // Handle device commands
          return handleDeviceExecute(req, res);
          
        default:
          console.log('Unknown intent:', input.intent);
          res.status(400).json({ error: 'Unknown intent' });
      }
    } else {
      // Handle direct API calls (for testing)
      return handleDirectAPI(req, res);
    }

  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

// Handle device sync (when Google Home discovers devices)
async function handleDeviceSync(req, res) {
  const response = {
    requestId: req.body.requestId,
    payload: {
      agentUserId: "user123",
      devices: [
        {
          id: "weavehacks-ai-device",
          type: "action.devices.types.CUSTOM_DEVICE",
          traits: [
            "action.devices.traits.OnOff",
            "action.devices.traits.TextResponse",
            "action.devices.traits.Modes"
          ],
          name: {
            defaultNames: ["WeaveHacks AI"],
            name: "WeaveHacks AI",
            nicknames: ["AI Assistant", "Assistant"]
          },
          deviceInfo: {
            manufacturer: "WeaveHacks",
            model: "AI Assistant v1.0",
            hwVersion: "1.0",
            swVersion: "1.0"
          },
          attributes: {
            availableModes: [
              {
                name: "mode",
                name_values: [
                  {
                    name_synonym: ["mode", "setting"],
                    lang: "en"
                  }
                ],
                settings: [
                  {
                    setting_name: "chat",
                    setting_values: [
                      {
                        setting_synonym: ["chat", "conversation", "talk"],
                        lang: "en"
                      }
                    ]
                  },
                  {
                    setting_name: "assist",
                    setting_values: [
                      {
                        setting_synonym: ["assist", "help", "support"],
                        lang: "en"
                      }
                    ]
                  }
                ],
                ordered: false
              }
            ]
          },
          willReportState: false,
          roomHint: "Living Room"
        }
      ]
    }
  };
  
  console.log('Device sync response:', JSON.stringify(response, null, 2));
  res.json(response);
}

// Handle device query (get device state)
async function handleDeviceQuery(req, res) {
  const response = {
    requestId: req.body.requestId,
    payload: {
      devices: {
        "weavehacks-ai-device": {
          online: true,
          on: true,
          currentModeSettings: {
            mode: "chat"
          }
        }
      }
    }
  };
  
  console.log('Device query response:', JSON.stringify(response, null, 2));
  res.json(response);
}

// Handle device execute (process commands)
async function handleDeviceExecute(req, res) {
  const commands = req.body.inputs[0].payload.commands;
  const results = [];
  
  for (const command of commands) {
    for (const execution of command.execution) {
      try {
        let userQuery = "";
        
        // Extract user query from different command types
        if (execution.command === "action.devices.commands.OnOff") {
          userQuery = execution.params.on ? "Turn on AI assistant" : "Turn off AI assistant";
        } else if (execution.command === "action.devices.commands.SetModes") {
          const mode = execution.params.updateModeSettings?.mode;
          userQuery = `Switch to ${mode} mode`;
        } else if (execution.command === "action.devices.commands.TextResponse") {
          userQuery = execution.params.text || "Hello";
        } else {
          // For any other command, try to extract text
          userQuery = execution.params?.text || execution.params?.query || "Hello";
        }
        
        console.log('Processing command:', execution.command, 'with query:', userQuery);
        
        // Call your /prompt API
        const apiResponse = await fetch('YOUR_API_URL/prompt', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_API_KEY'
          },
          body: JSON.stringify({
            query: userQuery,
            command: execution.command,
            timestamp: new Date().toISOString()
          })
        });
        
        if (!apiResponse.ok) {
          throw new Error(`API call failed: ${apiResponse.status}`);
        }
        
        const apiData = await apiResponse.json();
        console.log('API response:', apiData);
        
        // Add successful result
        results.push({
          ids: command.devices.map(d => d.id),
          status: "SUCCESS",
          states: {
            online: true,
            on: true,
            currentModeSettings: {
              mode: "chat"
            }
          }
        });
        
      } catch (error) {
        console.error('Error processing command:', error);
        results.push({
          ids: command.devices.map(d => d.id),
          status: "ERROR",
          errorCode: "deviceTurnedOff"
        });
      }
    }
  }
  
  const response = {
    requestId: req.body.requestId,
    payload: {
      commands: results
    }
  };
  
  console.log('Execute response:', JSON.stringify(response, null, 2));
  res.json(response);
}

// Handle direct API calls (for testing)
async function handleDirectAPI(req, res) {
  const userQuery = req.body.text || req.body.query || "Hello";
  
  try {
    const apiResponse = await fetch('YOUR_API_URL/prompt', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY'
      },
      body: JSON.stringify({
        query: userQuery,
        timestamp: new Date().toISOString()
      })
    });
    
    if (!apiResponse.ok) {
      throw new Error(`API call failed: ${apiResponse.status}`);
    }
    
    const apiData = await apiResponse.json();
    
    res.json({
      success: true,
      response: apiData.response || apiData.message || "I received your request",
      originalQuery: userQuery
    });
    
  } catch (error) {
    console.error('Error in direct API call:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
} 