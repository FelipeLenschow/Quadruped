package com.quadruped.console

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.ConnectivityManager
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.Process
import android.util.Log
import java.util.concurrent.locks.LockSupport

/**
 * Owns the DDS participant and the heartbeat thread. A foreground service so the
 * heartbeat keeps going with the screen off; if the process dies anyway, the
 * heartbeat stops and the robot's watchdog cuts it, as with a dead laptop console.
 */
class ConsoleService : Service() {
    companion object {
        private const val TAG = "QuadrupedConsole"
        private const val CHANNEL = "console"
        const val ACTION_START = "start"
        const val ACTION_STOP = "stop"

        @Volatile var core: ConsoleCore? = null
            private set
        @Volatile var linkInfo: String = "stopped"
            private set

        fun start(ctx: Context) = ctx.startForegroundService(Intent(ctx, ConsoleService::class.java).setAction(ACTION_START))
        fun stop(ctx: Context) = ctx.startService(Intent(ctx, ConsoleService::class.java).setAction(ACTION_STOP))
    }

    private val loops = mutableListOf<Thread>()
    @Volatile private var running = false
    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null
    private var multicastLock: WifiManager.MulticastLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                shutdown("console stopped")
                stopSelf()
            }
            else -> {
                startInForeground()
                if (!running) startLink()
            }
        }
        return START_NOT_STICKY
    }

    /** Swiping the app away is Ctrl-C on the console: e-stop, then exit. */
    override fun onTaskRemoved(rootIntent: Intent?) {
        shutdown("app closed")
        stopSelf()
    }

    override fun onDestroy() {
        shutdown("service destroyed")
        super.onDestroy()
    }

    private fun startLink() {
        val s = Settings.loadLink(this)
        val cm = getSystemService(ConnectivityManager::class.java)
        val (network, localIp) = RobotNetwork.pick(cm) ?: run {
            linkInfo = "no Wi-Fi/Ethernet network"
            stopSelf()
            return
        }
        // The robot AP has no internet, so Android may keep mobile data as the default
        // network. Binding the process makes every DDS socket go out on the robot's network.
        cm.bindProcessToNetwork(network)

        val useServer = when (s.discoveryMode) {
            "on" -> true
            "off" -> false
            else -> localIp.substringBeforeLast('.') == s.discoveryServer.substringBefore(':').substringBeforeLast('.')
        }
        val server = if (useServer) s.discoveryServer else ""
        acquireLocks(multicast = !useServer)

        NativeLink.start(s.domainId, server, localIp)?.let { err ->
            linkInfo = "DDS start failed: $err"
            Log.e(TAG, linkInfo)
            releaseLocks()
            cm.bindProcessToNetwork(null)
            stopSelf()
            return
        }
        ConsoleCore.PUBLICATIONS.forEach { (topic, type) -> NativeLink.advertise(topic, type) }
        ConsoleCore.SUBSCRIPTIONS.forEach { (topic, type) -> NativeLink.subscribe(topic, type) }
        val c = ConsoleCore(NativeLink, Settings.loadParams(this))
        core = c
        linkInfo = "domain ${s.domainId}, ${if (useServer) "server $server" else "multicast"}, local $localIp"
        Log.i(TAG, linkInfo)

        running = true
        periodic("heartbeat", s.heartbeatHz) { c.tick() }
        // twist_mux drops an input after 0.5 s; 20 Hz keeps well inside that.
        periodic("drive", 20.0) { c.tickDrive() }
    }

    private fun periodic(name: String, hz: Double, body: () -> Unit) {
        val periodNs = (1e9 / hz).toLong()
        loops += Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_DISPLAY)
            var next = System.nanoTime()
            while (running) {
                try {
                    body()
                } catch (e: Exception) {
                    Log.e(TAG, "$name failed", e)
                }
                next += periodNs
                val now = System.nanoTime()
                if (next < now) next = now
                LockSupport.parkNanos(next - now)
            }
        }, name).also { it.start() }
    }

    private fun shutdown(reason: String) {
        if (!running) return
        running = false
        loops.forEach { it.join(1000) }
        loops.clear()
        core?.let {
            Log.w(TAG, "E-STOP on shutdown: $reason")
            it.setDrive(false)
            it.emitEstop()
            Thread.sleep(100) // let the reliable writer flush before the participant goes
        }
        core = null
        NativeLink.stop()
        getSystemService(ConnectivityManager::class.java).bindProcessToNetwork(null)
        releaseLocks()
        linkInfo = "stopped ($reason)"
    }

    private fun acquireLocks(multicast: Boolean) {
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$TAG:heartbeat").apply { acquire() }
        val wm = applicationContext.getSystemService(WifiManager::class.java)
        // Wi-Fi power save adds 100-300 ms of latency, close to the 0.35 s watchdog.
        @Suppress("DEPRECATION")
        val mode = if (Build.VERSION.SDK_INT >= 29) WifiManager.WIFI_MODE_FULL_LOW_LATENCY
        else WifiManager.WIFI_MODE_FULL_HIGH_PERF
        wifiLock = wm.createWifiLock(mode, "$TAG:wifi").apply { acquire() }
        if (multicast) multicastLock = wm.createMulticastLock("$TAG:mcast").apply { acquire() }
    }

    private fun releaseLocks() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wifiLock?.takeIf { it.isHeld }?.release()
        multicastLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
        wifiLock = null
        multicastLock = null
    }

    private fun startInForeground() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Console", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
        )
        val n: Notification = Notification.Builder(this, CHANNEL)
            .setContentTitle("Quadruped console")
            .setContentText("Heartbeat running")
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentIntent(open)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)
        } else {
            startForeground(1, n)
        }
    }
}
