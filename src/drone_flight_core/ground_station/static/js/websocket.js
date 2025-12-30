/**
 * WebSocket Client for Ground Station
 *
 * Manages real-time connection to drone for telemetry,
 * video streams, and command acknowledgments.
 */

class WebSocketClient {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || `ws://${window.location.host}`;
        this.reconnectInterval = options.reconnectInterval || 3000;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 10;

        this.connections = {
            telemetry: null,
            video: null,
            commands: null
        };

        this.reconnectAttempts = {
            telemetry: 0,
            video: 0,
            commands: 0
        };

        this.handlers = {
            telemetry: [],
            video: [],
            alert: [],
            detection: [],
            command_ack: [],
            connection: []
        };

        this.isConnected = false;
    }

    /**
     * Connect to all WebSocket endpoints
     */
    connect() {
        this.connectTelemetry();
        this.connectVideo();
        this.connectCommands();
    }

    /**
     * Connect to telemetry WebSocket
     */
    connectTelemetry() {
        const url = `${this.baseUrl}/ws/telemetry`;
        console.log(`Connecting to telemetry: ${url}`);

        try {
            this.connections.telemetry = new WebSocket(url);

            this.connections.telemetry.onopen = () => {
                console.log('Telemetry WebSocket connected');
                this.reconnectAttempts.telemetry = 0;
                this.updateConnectionStatus();
            };

            this.connections.telemetry.onmessage = (event) => {
                this.handleMessage('telemetry', event.data);
            };

            this.connections.telemetry.onclose = () => {
                console.log('Telemetry WebSocket closed');
                this.scheduleReconnect('telemetry', this.connectTelemetry.bind(this));
            };

            this.connections.telemetry.onerror = (error) => {
                console.error('Telemetry WebSocket error:', error);
            };
        } catch (e) {
            console.error('Failed to connect telemetry:', e);
            this.scheduleReconnect('telemetry', this.connectTelemetry.bind(this));
        }
    }

    /**
     * Connect to video WebSocket
     */
    connectVideo() {
        const url = `${this.baseUrl}/ws/video`;
        console.log(`Connecting to video: ${url}`);

        try {
            this.connections.video = new WebSocket(url);
            this.connections.video.binaryType = 'arraybuffer';

            this.connections.video.onopen = () => {
                console.log('Video WebSocket connected');
                this.reconnectAttempts.video = 0;
                this.updateConnectionStatus();
            };

            this.connections.video.onmessage = (event) => {
                this.handleVideoMessage(event.data);
            };

            this.connections.video.onclose = () => {
                console.log('Video WebSocket closed');
                this.scheduleReconnect('video', this.connectVideo.bind(this));
            };

            this.connections.video.onerror = (error) => {
                console.error('Video WebSocket error:', error);
            };
        } catch (e) {
            console.error('Failed to connect video:', e);
            this.scheduleReconnect('video', this.connectVideo.bind(this));
        }
    }

    /**
     * Connect to commands WebSocket
     */
    connectCommands() {
        const url = `${this.baseUrl}/ws/commands`;
        console.log(`Connecting to commands: ${url}`);

        try {
            this.connections.commands = new WebSocket(url);

            this.connections.commands.onopen = () => {
                console.log('Commands WebSocket connected');
                this.reconnectAttempts.commands = 0;
                this.updateConnectionStatus();
            };

            this.connections.commands.onmessage = (event) => {
                this.handleMessage('command_ack', event.data);
            };

            this.connections.commands.onclose = () => {
                console.log('Commands WebSocket closed');
                this.scheduleReconnect('commands', this.connectCommands.bind(this));
            };

            this.connections.commands.onerror = (error) => {
                console.error('Commands WebSocket error:', error);
            };
        } catch (e) {
            console.error('Failed to connect commands:', e);
            this.scheduleReconnect('commands', this.connectCommands.bind(this));
        }
    }

    /**
     * Handle incoming telemetry/command messages
     */
    handleMessage(type, data) {
        try {
            const message = JSON.parse(data);

            // Route message to appropriate handlers
            if (message.type === 'telemetry') {
                this.emit('telemetry', message.data);
            } else if (message.type === 'alert') {
                this.emit('alert', message.data);
            } else if (message.type === 'detection') {
                this.emit('detection', message.data);
            } else if (message.type === 'command_ack') {
                this.emit('command_ack', message.data);
            } else if (message.type === 'video_info') {
                // Video metadata
                this.emit('video', { type: 'info', data: message.data });
            }
        } catch (e) {
            console.error('Failed to parse message:', e, data);
        }
    }

    /**
     * Handle incoming video frames
     */
    handleVideoMessage(data) {
        if (data instanceof ArrayBuffer) {
            // Binary frame data
            const blob = new Blob([data], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            this.emit('video', { type: 'frame', url: url });
        } else {
            // JSON metadata
            this.handleMessage('video', data);
        }
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect(type, connectFn) {
        this.updateConnectionStatus();

        if (this.reconnectAttempts[type] >= this.maxReconnectAttempts) {
            console.error(`Max reconnect attempts reached for ${type}`);
            return;
        }

        this.reconnectAttempts[type]++;
        const delay = this.reconnectInterval * Math.pow(1.5, this.reconnectAttempts[type] - 1);

        console.log(`Reconnecting ${type} in ${delay}ms (attempt ${this.reconnectAttempts[type]})`);
        setTimeout(connectFn, delay);
    }

    /**
     * Update overall connection status
     */
    updateConnectionStatus() {
        const telemetryConnected = this.connections.telemetry?.readyState === WebSocket.OPEN;
        const videoConnected = this.connections.video?.readyState === WebSocket.OPEN;
        const commandsConnected = this.connections.commands?.readyState === WebSocket.OPEN;

        const wasConnected = this.isConnected;
        this.isConnected = telemetryConnected && commandsConnected;

        if (wasConnected !== this.isConnected) {
            this.emit('connection', {
                connected: this.isConnected,
                telemetry: telemetryConnected,
                video: videoConnected,
                commands: commandsConnected
            });
        }
    }

    /**
     * Send command to drone
     */
    sendCommand(command, params = {}) {
        if (!this.connections.commands || this.connections.commands.readyState !== WebSocket.OPEN) {
            console.error('Commands WebSocket not connected');
            return false;
        }

        const message = {
            type: 'command',
            command: command,
            params: params,
            timestamp: new Date().toISOString()
        };

        this.connections.commands.send(JSON.stringify(message));
        console.log('Sent command:', command, params);
        return true;
    }

    /**
     * Request video stream from specific camera
     */
    requestVideoStream(camera = 'rgb', resolution = null, fps = null) {
        if (!this.connections.video || this.connections.video.readyState !== WebSocket.OPEN) {
            console.error('Video WebSocket not connected');
            return false;
        }

        const request = {
            type: 'stream_request',
            camera: camera,
            resolution: resolution,
            fps: fps
        };

        this.connections.video.send(JSON.stringify(request));
        return true;
    }

    /**
     * Register event handler
     */
    on(event, handler) {
        if (!this.handlers[event]) {
            this.handlers[event] = [];
        }
        this.handlers[event].push(handler);
    }

    /**
     * Remove event handler
     */
    off(event, handler) {
        if (this.handlers[event]) {
            this.handlers[event] = this.handlers[event].filter(h => h !== handler);
        }
    }

    /**
     * Emit event to handlers
     */
    emit(event, data) {
        if (this.handlers[event]) {
            this.handlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (e) {
                    console.error(`Handler error for ${event}:`, e);
                }
            });
        }
    }

    /**
     * Disconnect all WebSockets
     */
    disconnect() {
        Object.values(this.connections).forEach(conn => {
            if (conn) {
                conn.close();
            }
        });
        this.isConnected = false;
    }
}

// Global WebSocket client instance
const wsClient = new WebSocketClient();
