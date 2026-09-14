package `in`.greenroot.app

import android.annotation.SuppressLint
import android.app.DownloadManager
import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.webkit.CookieManager
import android.webkit.DownloadListener
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity

/**
 * A thin shell around the deployed GREENROOT dashboard.
 *
 * The recommendation model runs on the server, so this is deliberately a
 * WebView container rather than a reimplementation: one codebase, one set of
 * behaviour, and the phone gets fixes the moment the server is redeployed.
 *
 * Set [APP_URL] to your deployed address before building.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var web: WebView

    companion object {
        /** Your Streamlit Cloud URL. Must be https for cleartext to stay off. */
        private const val APP_URL = "https://greenroot.streamlit.app"
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        web = WebView(this)
        setContentView(web)

        web.settings.apply {
            javaScriptEnabled = true          // Streamlit is a JS application
            domStorageEnabled = true          // session state lives here
            loadWithOverviewMode = true
            useWideViewPort = true
            builtInZoomControls = true
            displayZoomControls = false
            mediaPlaybackRequiresUserGesture = false
        }
        CookieManager.getInstance().setAcceptThirdPartyCookies(web, true)

        // Keep navigation inside the app; hand anything external to the browser.
        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView,
                request: WebResourceRequest,
            ): Boolean {
                val host = request.url.host ?: return false
                val ours = Uri.parse(APP_URL).host ?: return false
                if (host == ours) return false
                startActivity(android.content.Intent(
                    android.content.Intent.ACTION_VIEW, request.url))
                return true
            }

            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: android.webkit.WebResourceError,
            ) {
                if (request.isForMainFrame) showOfflineNotice()
            }
        }

        // The Soil Health Card exports are downloads; route them to the
        // system download manager so they land in the phone's Downloads.
        web.setDownloadListener(DownloadListener { url, _, _, mimeType, _ ->
            try {
                val request = DownloadManager.Request(Uri.parse(url)).apply {
                    setMimeType(mimeType)
                    setNotificationVisibility(
                        DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                    setDestinationInExternalPublicDir(
                        android.os.Environment.DIRECTORY_DOWNLOADS,
                        URLUtilName(url, mimeType))
                }
                (getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager)
                    .enqueue(request)
                Toast.makeText(this, "Saving to Downloads…", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Could not save the file", Toast.LENGTH_SHORT).show()
            }
        })

        // Back button walks the web history before leaving the app.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (web.canGoBack()) web.goBack() else finish()
            }
        })

        if (isOnline()) web.loadUrl(APP_URL) else showOfflineNotice()
    }

    private fun URLUtilName(url: String, mimeType: String?): String =
        android.webkit.URLUtil.guessFileName(url, null, mimeType)

    private fun isOnline(): Boolean {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    /** The app needs the server, so say so plainly rather than showing a blank page. */
    private fun showOfflineNotice() {
        web.loadDataWithBaseURL(
            null,
            """
            <html><head><meta name="viewport"
              content="width=device-width, initial-scale=1"></head>
            <body style="font-family:sans-serif;margin:0;display:flex;
                         align-items:center;justify-content:center;height:100vh;
                         background:#f4f8f5;color:#14281d;text-align:center">
              <div style="padding:28px;max-width:22rem">
                <div style="font-size:52px">📶</div>
                <h2 style="margin:12px 0 8px;color:#1f7a4d">No internet</h2>
                <p style="line-height:1.6;color:#4a5b51">
                  GreenRoot needs a connection to work out your crop advice.
                  Turn on mobile data or Wi-Fi, then reopen the app.
                </p>
              </div>
            </body></html>
            """.trimIndent(),
            "text/html", "utf-8", null,
        )
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        web.saveState(outState)
    }
}
