// App.js - Toddler Alert Mobile App
import React, { useState, useEffect, useRef } from 'react';
import { 
    StyleSheet, 
    Text, 
    View, 
    TouchableOpacity, 
    Modal, 
    SafeAreaView, 
    AppState, 
    Platform,
    Alert,
    Vibration,
    StatusBar
} from 'react-native';
import * as Notifications from 'expo-notifications';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Audio } from 'expo-av';
import { MaterialIcons } from '@expo/vector-icons';
import { io } from 'socket.io-client';

// Dark theme styling similar to desktop app
const DarkThemeStyle = {
    PRIMARY_COLOR: "#2979FF",      // Vibrant blue
    SECONDARY_COLOR: "#5C6BC0",    // Indigo
    WARNING_COLOR: "#FF5252",      // Bright red for warnings
    SUCCESS_COLOR: "#66BB6A",      // Green for success
    BACKGROUND_COLOR: "#1E1E2E",   // Dark deep blue/purple background
    CARD_COLOR: "#2A2A3C",         // Slightly lighter card background
    PANEL_COLOR: "#252536",        // Medium dark for panels
    TEXT_PRIMARY: "#FFFFFF",       // White for primary text
    TEXT_SECONDARY: "#B0B0C0",     // Light gray/lavender for secondary text
    ACCENT_COLOR: "#BB86FC",       // Purple accent
};

// Set up notification handler
Notifications.setNotificationHandler({
    handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: true,
        shouldSetBadge: true,
    }),
});

export default function App() {
    // State variables
    const [connected, setConnected] = useState(false);
    const [serverAddress, setServerAddress] = useState('http://192.168.254.110:3000');
    const [socket, setSocket] = useState(null);
    const [currentAlert, setCurrentAlert] = useState(null);
    const [sound, setSound] = useState(null);
    const [showMenu, setShowMenu] = useState(false);
    const [showAbout, setShowAbout] = useState(false);
    const appState = useRef(AppState.currentState);
    const notificationListener = useRef();
    const responseListener = useRef();
    const socketTimeoutRef = useRef(null);
    const connectionTimeoutMs = 15000; // 15 seconds timeout

    useEffect(() => {
        let mounted = true;
        
        async function initialize() {
            try {
                // Request notification permissions
                await registerForPushNotificationsAsync();
                
                if (mounted) {
                    connectToServer(serverAddress);
                }
            } catch (error) {
                console.error('Error during app initialization', error);
                Alert.alert(
                    'Initialization Error',
                    'There was a problem starting the app. Please restart the app.'
                );
            }
        }
        
        initialize();
        
        // Set up notification listeners
        notificationListener.current = Notifications.addNotificationReceivedListener(notification => {
            if (mounted) handleAlertNotification(notification.request.content.data);
        });
        
        responseListener.current = Notifications.addNotificationResponseReceivedListener(response => {
            if (currentAlert && mounted) {
                stopAlarm();
            }
        });
        
        // Monitor app state changes for background/foreground transitions
        const subscription = AppState.addEventListener('change', nextAppState => {
            if (appState.current.match(/inactive|background/) && nextAppState === 'active') {
                if (socket && !socket.connected && mounted) {
                    reconnectSocket();
                }
            }
            appState.current = nextAppState;
        });
        
        return () => {
            mounted = false;
            Notifications.removeNotificationSubscription(notificationListener.current);
            Notifications.removeNotificationSubscription(responseListener.current);
            subscription.remove();
            if (socketTimeoutRef.current) clearTimeout(socketTimeoutRef.current);
            if (socket) socket.disconnect();
            if (sound) {
                (async () => {
                    try {
                        await sound.stopAsync();
                        await sound.unloadAsync();
                    } catch (error) {
                        console.error('Error unloading sound', error);
                    }
                })();
            }
            Vibration.cancel();
        };
    }, []);
    
    // Play alarm sound with error handling
    const playAlarmSound = async () => {
        try {
            // Unload any existing sound first
            if (sound) {
                try {
                    await sound.stopAsync();
                    await sound.unloadAsync();
                } catch (error) {
                    console.error('Error stopping previous sound', error);
                }
            }
            
            // Create sound object
            const soundObject = new Audio.Sound();
            
            try {
                // Try to load the sound file if it exists
                await soundObject.loadAsync(require('D:\\vscode\\Python\\Thesis\\thesis-toddler-monitoring-system\\gui\\toddler-monitoring-system\\ToddlerAlert\\assets\\alert.wav'));
            } catch (error) {
                console.log('Using fallback sound - vibration');
                // If the sound file doesn't exist, use vibration
                if (Platform.OS === 'android') {
                    Vibration.vibrate([500, 500, 500, 500, 500, 500], true);
                }
                return; // Exit without setting the sound
            }
            
            await soundObject.setIsLoopingAsync(true);
            await soundObject.playAsync();
            setSound(soundObject);
            
            // Start vibration pattern
            if (Platform.OS === 'android') {
                const PATTERN = [1000, 2000, 3000];
                Vibration.vibrate(PATTERN, true);
            }
        } catch (error) {
            console.error('Error playing sound', error);
            // Just vibrate if sound fails
            if (Platform.OS === 'android') {
                Vibration.vibrate([500, 500, 500, 500, 500, 500], true);
            }
        }
    };
    
    // Stop alarm sound
    const stopAlarm = async () => {
        try {
            if (sound) {
                await sound.stopAsync();
                await sound.unloadAsync();
                setSound(null);
            }
            setCurrentAlert(null);
            Vibration.cancel();
        } catch (error) {
            console.error('Error stopping sound', error);
            // Force reset sound state even if there was an error
            setSound(null);
            setCurrentAlert(null);
            Vibration.cancel();
        }
    };
    
    // Handle receiving alert from server
    const handleAlertNotification = async (alertData) => {
        if (!alertData) {
            console.error('Received empty alert data');
            return;
        }
        
        try {
            // Create new alert
            const newAlert = {
                ...alertData,
                timestamp: new Date().toISOString()
            };
            
            // Set current alert
            setCurrentAlert(newAlert);
            
            // Play alarm sound
            playAlarmSound();
        } catch (error) {
            console.error('Error handling alert notification', error);
        }
    };
    
    // Connect to server using provided address
    const connectToServer = (address) => {
        if (!address) {
            Alert.alert(
                'Connection Error',
                'Invalid server address.',
                [{ text: 'OK' }]
            );
            return;
        }
        
        try {
            // Clear any existing timeout
            if (socketTimeoutRef.current) {
                clearTimeout(socketTimeoutRef.current);
            }
            
            // Set connection status to indicate attempting connection
            setConnected(false);
            
            const newSocket = io(address, {
                reconnectionAttempts: 5,
                timeout: connectionTimeoutMs,
                reconnectionDelay: 1000,
                forceNew: true,
            });
            
            // Set timeout for connection
            socketTimeoutRef.current = setTimeout(() => {
                if (newSocket && !newSocket.connected) {
                    newSocket.disconnect();
                    Alert.alert(
                        'Connection Timeout',
                        'Failed to connect to server. The app will retry automatically.',
                        [{ text: 'OK' }]
                    );
                }
            }, connectionTimeoutMs);
            
            newSocket.on('connect', () => {
                clearTimeout(socketTimeoutRef.current);
                setConnected(true);
                setSocket(newSocket);
                
                // Register device with server
                newSocket.emit('register_mobile', {
                    deviceId: getDeviceId(),
                    platform: Platform.OS,
                    version: '1.0.0'
                });
            });
            
            newSocket.on('connect_error', (error) => {
                console.error('Socket connection error', error);
                setConnected(false);
            });
            
            newSocket.on('disconnect', () => {
                console.log('Socket disconnected');
                setConnected(false);
            });
            
            newSocket.on('toddler_alert', (alertData) => {
                schedulePushNotification(alertData);
                handleAlertNotification(alertData);
            });
            
            setSocket(newSocket);
        } catch (error) {
            console.error('Socket connection error:', error);
            Alert.alert(
                'Connection Error',
                'Failed to connect to the monitoring system. Will retry in a few seconds.',
                [{ text: 'OK' }]
            );
            
            // Retry connection after a delay
            setTimeout(() => {
                reconnectSocket();
            }, 5000);
        }
    };
    
    // Reconnect socket if connection lost
    const reconnectSocket = () => {
        if (serverAddress) {
            connectToServer(serverAddress);
        }
    };
    
    // Get unique device ID
    const getDeviceId = () => {
        return `${Platform.OS}_${Math.random().toString(36).substring(2, 10)}_${Date.now()}`;
    };
    
    // Schedule push notification
    async function schedulePushNotification(alertData) {
        if (!alertData) {
            console.error('Cannot schedule notification with empty data');
            return;
        }
        
        try {
            const alertType = alertData.type || 'danger';
            const alertMessage = alertData.message || 'Toddler alert detected!';
            
            await Notifications.scheduleNotificationAsync({
                content: {
                    title: alertType === 'geofence' ? 'Geofence Alert!' : 'Hazard Alert!',
                    body: alertMessage,
                    data: alertData,
                    sound: 'default',
                    priority: 'high',
                },
                trigger: null, // Immediate notification
            });
        } catch (error) {
            console.error('Error scheduling notification:', error);
        }
    }
    
    // Register for push notifications
    async function registerForPushNotificationsAsync() {
        try {
            if (Platform.OS === 'android') {
                await Notifications.setNotificationChannelAsync('toddler-alerts', {
                    name: 'Toddler Alerts',
                    importance: Notifications.AndroidImportance.MAX,
                    vibrationPattern: [0, 250, 250, 250],
                    lightColor: '#FF453A',
                });
            }
            
            const { status: existingStatus } = await Notifications.getPermissionsAsync();
            let finalStatus = existingStatus;
            
            if (existingStatus !== 'granted') {
                const { status } = await Notifications.requestPermissionsAsync();
                finalStatus = status;
            }
            
            if (finalStatus !== 'granted') {
                Alert.alert(
                    'Permission Required',
                    'Push notifications are required for toddler alerts. Please enable them in your device settings.',
                    [{ text: 'OK' }]
                );
            }
        } catch (error) {
            console.error('Error registering for push notifications:', error);
            Alert.alert(
                'Notification Error',
                'Could not set up notifications. Some features may not work correctly.',
                [{ text: 'OK' }]
            );
        }
    }

    // About Dialog component
    const AboutDialog = () => (
        <Modal
            visible={showAbout}
            animationType="slide"
            transparent={true}
            onRequestClose={() => setShowAbout(false)}
        >
            <View style={styles.aboutModal}>
                <View style={styles.aboutContainer}>
                    <View style={styles.aboutHeader}>
                        <Text style={styles.aboutTitle}>About Toddler Alert</Text>
                    </View>
                    
                    <View style={styles.aboutBody}>
                        <View style={styles.logoPlaceholder}>
                            <MaterialIcons name="child-care" size={60} color={DarkThemeStyle.PRIMARY_COLOR} />
                        </View>
                        <Text style={styles.aboutVersion}>Version 1.0.0</Text>
                        <Text style={styles.aboutDescription}>
                            Toddler Alert is a mobile companion app for the Toddler Monitoring System. 
                            It receives real-time alerts when your toddler is near hazardous objects or 
                            leaves a designated safe area.
                        </Text>
                        <Text style={styles.aboutCopyright}>© 2025 Toddler Safety Systems Inc.</Text>
                    </View>
                    
                    <TouchableOpacity
                        style={styles.closeButton}
                        onPress={() => setShowAbout(false)}
                    >
                        <Text style={styles.closeButtonText}>Close</Text>
                    </TouchableOpacity>
                </View>
            </View>
        </Modal>
    );
    
    // Render active alert modal
    const renderActiveAlert = () => {
        if (!currentAlert) return null;
        
        return (
            <Modal
                visible={!!currentAlert}
                animationType="slide"
                transparent={true}
                onRequestClose={() => {}}
            >
                <View style={styles.alertModal}>
                    <View style={styles.alertContainer}>
                        <View style={[
                            styles.alertHeader,
                            currentAlert.type === 'geofence' ? styles.geofenceHeader : styles.hazardHeader
                        ]}>
                            <MaterialIcons
                                name={currentAlert.type === 'geofence' ? 'location-off' : 'warning'}
                                size={40}
                                color="white"
                            />
                            <Text style={styles.alertTitle}>
                                {currentAlert.type === 'geofence' ? 'Geofence Alert!' : 'Hazard Alert!'}
                            </Text>
                        </View>
                        
                        <View style={styles.alertBody}>
                            <Text style={styles.alertMessage}>{currentAlert.message}</Text>
                            <Text style={styles.alertTime}>
                                {new Date(currentAlert.timestamp).toLocaleString()}
                            </Text>
                        </View>
                        
                        <TouchableOpacity
                            style={styles.acknowledgeButton}
                            onPress={stopAlarm}
                        >
                            <Text style={styles.acknowledgeText}>Acknowledge</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>
        );
    };

    // Menu component
    const DropdownMenu = () => (
        <Modal
            visible={showMenu}
            animationType="fade"
            transparent={true}
            onRequestClose={() => setShowMenu(false)}
        >
            <TouchableOpacity 
                style={styles.menuOverlay}
                activeOpacity={1}
                onPress={() => setShowMenu(false)}
            >
                <View style={styles.menuContainer}>
                    <TouchableOpacity
                        style={styles.menuItem}
                        onPress={() => {
                            setShowMenu(false);
                            setShowAbout(true);
                        }}
                    >
                        <MaterialIcons name="info" size={24} color={DarkThemeStyle.TEXT_PRIMARY} />
                        <Text style={styles.menuText}>About</Text>
                    </TouchableOpacity>
                    
                    <TouchableOpacity
                        style={styles.menuItem}
                        onPress={() => {
                            setShowMenu(false);
                            reconnectSocket();
                        }}
                    >
                        <MaterialIcons name="refresh" size={24} color={DarkThemeStyle.TEXT_PRIMARY} />
                        <Text style={styles.menuText}>Reconnect</Text>
                    </TouchableOpacity>
                    
                    {__DEV__ && (
                        <TouchableOpacity
                            style={styles.menuItem}
                            onPress={() => {
                                setShowMenu(false);
                                // Simulate receiving an alert
                                handleAlertNotification({
                                    type: Math.random() > 0.5 ? 'geofence' : 'hazard',
                                    message: Math.random() > 0.5
                                        ? 'Toddler has left the safe area!' 
                                        : 'Toddler is near a hazard!',
                                    location: 'Kitchen',
                                    severity: 'high'
                                });
                            }}
                        >
                            <MaterialIcons name="bug-report" size={24} color={DarkThemeStyle.TEXT_PRIMARY} />
                            <Text style={styles.menuText}>Test Alert</Text>
                        </TouchableOpacity>
                    )}
                </View>
            </TouchableOpacity>
        </Modal>
    );
    
    // Main render function
    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor={DarkThemeStyle.PANEL_COLOR} />
            
            {/* Header */}
            <View style={styles.header}>
                <View style={styles.titleContainer}>
                    <MaterialIcons name="child-care" size={28} color="white" style={styles.titleIcon} />
                    <Text style={styles.headerText}>Toddler Alert</Text>
                </View>
                
                <TouchableOpacity
                    style={styles.menuButton}
                    onPress={() => setShowMenu(true)}
                >
                    <MaterialIcons name="more-vert" size={24} color="white" />
                </TouchableOpacity>
            </View>
            
            {/* Status indicator */}
            <View style={styles.statusBar}>
                <View style={styles.statusIndicator}>
                    <View style={[
                        styles.statusDot,
                        connected ? styles.statusConnected : styles.statusDisconnected
                    ]} />
                    <Text style={styles.statusText}>
                        {connected ? 'Connected' : 'Reconnecting...'}
                    </Text>
                </View>
            </View>
            
            {/* Main content */}
            <View style={styles.content}>
                <View style={styles.mainContent}>
                    <MaterialIcons
                        name={connected ? "child-care" : "cloud-off"}
                        size={80}
                        color={DarkThemeStyle.PRIMARY_COLOR}
                    />
                    
                    <Text style={styles.mainText}>
                        {connected 
                            ? "Ready to Receive Alerts" 
                            : "Connecting to Alert System..."}
                    </Text>
                    
                    <Text style={styles.subText}>
                        This app will notify you when your toddler is near a hazard or leaves the safe area.
                    </Text>
                    
                    {!connected && (
                        <TouchableOpacity
                            style={styles.reconnectButton}
                            onPress={reconnectSocket}
                        >
                            <Text style={styles.reconnectText}>Try Reconnecting</Text>
                        </TouchableOpacity>
                    )}
                    
                    {/* For development testing only */}
                    {__DEV__ && (
                        <TouchableOpacity
                            style={[styles.reconnectButton, {marginTop: 20, backgroundColor: DarkThemeStyle.WARNING_COLOR}]}
                            onPress={() => handleAlertNotification({
                                type: 'hazard',
                                message: 'Test alert: Toddler near hazard!',
                                location: 'Living Room',
                                severity: 'high'
                            })}
                        >
                            <Text style={styles.reconnectText}>Test Alert (DEV)</Text>
                        </TouchableOpacity>
                    )}
                </View>
            </View>
            
            {/* Modals */}
            {renderActiveAlert()}
            <DropdownMenu />
            <AboutDialog />
        </SafeAreaView>
    );
}

// Styles
const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: DarkThemeStyle.BACKGROUND_COLOR,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 12,
        backgroundColor: DarkThemeStyle.PANEL_COLOR,
        elevation: 4,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 3,
    },
    titleContainer: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    titleIcon: {
        marginRight: 8,
    },
    headerText: {
        color: DarkThemeStyle.TEXT_PRIMARY,
        fontSize: 20,
        fontWeight: 'bold',
    },
    menuButton: {
        padding: 8,
    },
    statusBar: {
        backgroundColor: DarkThemeStyle.CARD_COLOR,
        paddingVertical: 8,
        paddingHorizontal: 16,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(255,255,255,0.1)',
    },
    statusIndicator: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    statusDot: {
        width: 10,
        height: 10,
        borderRadius: 5,
        marginRight: 6,
    },
    statusConnected: {
        backgroundColor: DarkThemeStyle.SUCCESS_COLOR,
    },
    statusDisconnected: {
        backgroundColor: DarkThemeStyle.WARNING_COLOR,
    },
    statusText: {
        color: DarkThemeStyle.TEXT_SECONDARY,
        fontSize: 14,
    },
    content: {
        flex: 1,
        padding: 20,
    },
    mainContent: {
        backgroundColor: DarkThemeStyle.CARD_COLOR,
        borderRadius: 12,
        padding: 24,
        alignItems: 'center',
        justifyContent: 'center',
        marginTop: 20,
        elevation: 2,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 3,
        flex: 1,
    },
    mainText: {
        color: DarkThemeStyle.TEXT_PRIMARY,
        fontSize: 22,
        fontWeight: 'bold',
        textAlign: 'center',
        marginTop: 20,
        marginBottom: 10,
    },
    subText: {
        color: DarkThemeStyle.TEXT_SECONDARY,
        fontSize: 16,
        textAlign: 'center',
        marginBottom: 30,
        paddingHorizontal: 20,
    },
    reconnectButton: {
        backgroundColor: DarkThemeStyle.PRIMARY_COLOR,
        paddingVertical: 12,
        paddingHorizontal: 24,
        borderRadius: 25,
        alignItems: 'center',
        marginTop: 10,
    },
    reconnectText: {
        color: 'white',
        fontSize: 16,
        fontWeight: 'bold',
    },
    alertModal: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.8)',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    alertContainer: {
        backgroundColor: DarkThemeStyle.CARD_COLOR,
        borderRadius: 15,
        width: '90%',
        overflow: 'hidden',
        elevation: 8,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.5,
        shadowRadius: 6,
    },
    alertHeader: {
        padding: 20,
        alignItems: 'center',
        justifyContent: 'center',
    },
    hazardHeader: {
        backgroundColor: DarkThemeStyle.WARNING_COLOR,
    },
    geofenceHeader: {
        backgroundColor: '#FF9800',
    },
    alertTitle: {
        color: 'white',
        fontSize: 24,
        fontWeight: 'bold',
        marginTop: 10,
    },
    alertBody: {
        padding: 20,
        alignItems: 'center',
    },
    alertMessage: {
        color: DarkThemeStyle.TEXT_PRIMARY,
        fontSize: 18,
        textAlign: 'center',
        marginBottom: 15,
    },
    alertTime: {
        color: DarkThemeStyle.TEXT_SECONDARY,
        fontSize: 14,
    },
    acknowledgeButton: {
        backgroundColor: DarkThemeStyle.PRIMARY_COLOR,
        paddingVertical: 15,
        alignItems: 'center',
        marginTop: 10,
    },
    acknowledgeText: {
        color: 'white',
        fontSize: 18,
        fontWeight: 'bold',
    },
    menuOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
    },
    menuContainer: {
        position: 'absolute',
        top: 60,
        right: 16,
        backgroundColor: DarkThemeStyle.PANEL_COLOR,
        borderRadius: 8,
        elevation: 6,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.3,
        shadowRadius: 4,
        minWidth: 180,
    },
    menuItem: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 16,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(255,255,255,0.1)',
    },
    menuText: {
        color: DarkThemeStyle.TEXT_PRIMARY,
        fontSize: 16,
        marginLeft: 16,
    },
    aboutModal: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.8)',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    aboutContainer: {
        backgroundColor: DarkThemeStyle.CARD_COLOR,
        borderRadius: 15,
        width: '90%',
        overflow: 'hidden',
        elevation: 8,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.5,
        shadowRadius: 6,
    },
    aboutHeader: {
        backgroundColor: DarkThemeStyle.PANEL_COLOR,
        padding: 16,
        alignItems: 'center',
    },
    aboutTitle: {
        color: DarkThemeStyle.TEXT_PRIMARY,
        fontSize: 20,
        fontWeight: 'bold',
    },
    aboutBody: {
        padding: 24,
        alignItems: 'center',
    },
    logoPlaceholder: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: DarkThemeStyle.PANEL_COLOR,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 20,
    },
    aboutVersion: {
        color: DarkThemeStyle.PRIMARY_COLOR,
        fontSize: 16,
        fontWeight: 'bold',
        marginBottom: 16,
    },
    aboutDescription: {
        color: DarkThemeStyle.TEXT_PRIMARY,
        fontSize: 16,
        textAlign: 'center',
        marginBottom: 20,
        lineHeight: 24,
    },
    aboutCopyright: {
        color: DarkThemeStyle.TEXT_SECONDARY,
        fontSize: 14,
        marginTop: 20,
    },
    closeButton: {
        backgroundColor: DarkThemeStyle.PRIMARY_COLOR,
        paddingVertical: 15,
        alignItems: 'center',
        marginTop: 10,
    },
    closeButtonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: 'bold',
    },
});