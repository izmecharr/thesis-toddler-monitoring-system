const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",  // Allow connections from anywhere (for development)
    methods: ["GET", "POST"]
  }
});

const PORT = 3000;

// Basic route for testing
app.get('/', (req, res) => {
  res.send('Socket.IO server is running!');
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('A client connected:', socket.id);
  
  // Handle device registration
  socket.on('register_mobile', (deviceInfo) => {
    console.log('Mobile device registered:', deviceInfo);
    // Store device info for later use
    socket.deviceInfo = deviceInfo;
    socket.isMobile = true;
    
    // Acknowledge the registration
    socket.emit('registration_successful', { status: 'connected' });
  });
  
  // IMPORTANT: Handle alerts from the desktop tester directly
  socket.on('toddler_alert', (alertData) => {
    console.log('Received direct alert to forward:', alertData);
    // Forward to all mobile clients
    broadcastToMobiles(alertData);
  });
  
  // Handle alert requests from the tester
  socket.on('send_test_alert', (alertData) => {
    console.log('Received test alert request:', alertData);
    // Forward to all mobile clients
    broadcastToMobiles(alertData);
  });
  
  // Handle broadcast requests
  socket.on('broadcast_alert', (data) => {
    console.log('Received broadcast alert:', data.alert);
    broadcastToMobiles(data.alert);
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
  
  // Send a welcome message
  socket.emit('server_message', { message: 'Connected to Toddler Alert server!' });
});

// Helper function to broadcast to all mobile clients
function broadcastToMobiles(alertData) {
  // Count of mobile clients that received the alert
  let mobileCount = 0;
  
  // Try to send to all clients that identified as mobile
  io.sockets.sockets.forEach(clientSocket => {
    if (clientSocket.isMobile) {
      clientSocket.emit('toddler_alert', alertData);
      mobileCount++;
      console.log('Alert sent to mobile client:', clientSocket.id);
    }
  });
  
  // If no mobile clients were found, broadcast to everyone
  if (mobileCount === 0) {
    console.log('No mobile clients found, broadcasting to all clients');
    io.emit('toddler_alert', alertData);
  }
  
  console.log(`Alert broadcasted to ${mobileCount || 'all'} clients`);
}

// Start the server
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Socket.IO server running at http://0.0.0.0:${PORT}`);
  console.log(`Server IP for connection: http://YOUR_IP_ADDRESS:${PORT}`);
  console.log('Replace YOUR_IP_ADDRESS with your actual IP address');
});