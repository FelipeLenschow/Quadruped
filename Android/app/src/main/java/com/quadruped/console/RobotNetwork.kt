package com.quadruped.console

import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import java.net.Inet4Address

/**
 * The Wi-Fi (or Ethernet) network the robot is on, and our IPv4 address there. The
 * robot's AP has no internet, so Android may keep mobile data as the default network;
 * everything that talks to the robot goes through this one explicitly.
 */
object RobotNetwork {
    fun pick(cm: ConnectivityManager): Pair<Network, String>? {
        @Suppress("DEPRECATION")
        for (n in cm.allNetworks) {
            val caps = cm.getNetworkCapabilities(n) ?: continue
            if (!caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) &&
                !caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
            ) continue
            val ip = cm.getLinkProperties(n)?.linkAddresses
                ?.map { it.address }
                ?.firstOrNull { it is Inet4Address && !it.isLoopbackAddress }
                ?.hostAddress ?: continue
            return n to ip
        }
        return null
    }
}
