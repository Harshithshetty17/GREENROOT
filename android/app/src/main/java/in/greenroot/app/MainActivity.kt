package `in`.greenroot.app

import android.annotation.SuppressLint
import android.app.DownloadManager
import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.webkit.CookieManager
import android.webkit.DownloadListener
import android.webkit.URLUtil
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout

/**
 * GREENROOT on Android.
 *
 * The stacking ensemble runs on the server, so the recommendation screen is
 * the deployed dashboard in a WebView — one implementation of the model, and
 * the phone picks up fixes the moment the server is redeployed.
 *
 * What is native, and deliberately so: every recommendation the farmer saves
 * is written to the phone, and those saved cards are readable with no
 * connection at all. Fields do not have signal. A farmer who can only read
 * their fertiliser plan when the network is up has not been helped.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var web: WebView
    private lateinit var refresh: SwipeRefreshLayout
    private lateinit var store: CardStore

    /** True while the WebView is showing locally generated HTML. */
    private var showingLocalPage = false

    companion object {
        /**
         * The deployed dashboard. Must be https: cleartext is off in the
         * manifest, and Play rejects cleartext traffic to a production host.
         */
        private const val APP_URL = "https://iv3whf7b8pki4eqmuyac44.streamlit.app"
        private const val BRIDGE_NAME = "GreenRootNative"
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        // Must precede super.onCreate, or the system shows its own blank
        // window first and the launch reads as a stall.
        installSplashScreen()
        super.onCreate(savedInstanceState)

        store = CardStore(this)
        web = WebView(this)
        refresh = SwipeRefreshLayout(this).apply {
            setColorSchemeColors(0xFF1F7A4D.toInt())
            addView(web)
            setOnRefreshListener { reload() }
        }
        setContentView(refresh)

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
        attachBridge()

        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView,
                request: WebResourceRequest,
            ): Boolean {
                val host = request.url.host ?: return false
                if (host == Uri.parse(APP_URL).host) return false
                // addJavascriptInterface exposes the bridge to whatever page
                // is loaded, so nothing outside our own origin gets it: hand
                // foreign links to the browser instead of following them.
                startActivity(Intent(Intent.ACTION_VIEW, request.url))
                return true
            }

            override fun onPageFinished(view: WebView, url: String?) {
                refresh.isRefreshing = false
            }

            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: WebResourceError,
            ) {
                if (request.isForMainFrame) showOfflineLibrary()
            }
        }

        // Soil Health Card exports are downloads; route them to the system
        // download manager so they land in the phone's Downloads folder.
        web.setDownloadListener(DownloadListener { url, _, _, mimeType, _ ->
            try {
                val request = DownloadManager.Request(Uri.parse(url)).apply {
                    setMimeType(mimeType)
                    setNotificationVisibility(
                        DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                    setDestinationInExternalPublicDir(
                        Environment.DIRECTORY_DOWNLOADS,
                        URLUtil.guessFileName(url, null, mimeType))
                }
                (getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager)
                    .enqueue(request)
                Toast.makeText(this, "Saving to Downloads…", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Could not save the file", Toast.LENGTH_SHORT).show()
            }
        })

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                when {
                    // Backing out of the offline library returns to a live
                    // attempt rather than closing the app outright.
                    showingLocalPage && isOnline() -> reload()
                    !showingLocalPage && web.canGoBack() -> web.goBack()
                    else -> finish()
                }
            }
        })

        val restored = savedInstanceState?.let { web.restoreState(it) } != null
        if (!restored) reload()
    }

    private fun attachBridge() {
        web.addJavascriptInterface(
            NativeBridge(this, store) {
                runOnUiThread {
                    Toast.makeText(this, "Saved to this phone", Toast.LENGTH_SHORT).show()
                }
            },
            BRIDGE_NAME,
        )
    }

    private fun reload() {
        if (isOnline()) {
            showingLocalPage = false
            web.loadUrl(APP_URL)
        } else {
            showOfflineLibrary()
        }
    }

    private fun isOnline(): Boolean {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    /**
     * With no connection, show what the phone already holds.
     *
     * A blank page or a bare "no internet" notice would be the honest state
     * of the WebView but the wrong answer for the user: the advice they saved
     * is on this device and is exactly what they walked into the field to
     * read.
     */
    private fun showOfflineLibrary() {
        showingLocalPage = true
        refresh.isRefreshing = false
        val cards = store.all()

        val body = if (cards.isEmpty()) {
            """
            <div class="empty">
              <div class="glyph">🌱</div>
              <h2>Nothing saved yet</h2>
              <p>GreenRoot needs a connection to work out new advice. Once you
                 save a recommendation it stays on this phone, and you can read
                 it here with no signal at all.</p>
            </div>
            """
        } else {
            cards.joinToString("") { card ->
                val advice = card.advice.joinToString("") {
                    "<li>${esc(it)}</li>"
                }
                """
                <article class="card">
                  <div class="when">${esc(card.savedOn())}${
                      if (card.district.isBlank()) "" else " · " + esc(card.district)}</div>
                  <h3>${esc(card.crop)}</h3>
                  <div class="match">${card.confidence.toInt()} out of 100 match</div>
                  <div class="reads">${esc(card.readings)}</div>
                  ${if (advice.isEmpty()) "" else "<ul>$advice</ul>"}
                  <button class="share" data-id="${esc(card.id)}">
                    Share this advice
                  </button>
                </article>
                """
            }
        }

        web.loadDataWithBaseURL(
            null,
            """
            <html><head><meta name="viewport"
              content="width=device-width, initial-scale=1">
            <style>
              *{box-sizing:border-box}
              body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
                   margin:0;background:#f4f8f5;color:#14281d;padding:16px 16px 32px}
              header{background:linear-gradient(135deg,#115c3c,#2f9e5f);color:#fff;
                     border-radius:14px;padding:16px 18px;margin-bottom:14px}
              header h1{margin:0;font-size:19px;letter-spacing:.4px}
              header p{margin:4px 0 0;font-size:12.5px;opacity:.9}
              .card{background:#fff;border:1px solid #dfe8e2;border-radius:14px;
                    padding:16px 18px;margin-bottom:12px}
              .when{font-size:11.5px;letter-spacing:.5px;text-transform:uppercase;
                    color:#5c6f63;font-weight:600}
              .card h3{margin:4px 0 2px;font-size:24px;color:#14603c;
                       text-transform:uppercase;letter-spacing:.3px}
              .match{font-size:13.5px;color:#3d5548;font-weight:500}
              .reads{font-size:13px;color:#5c6f63;margin-top:6px}
              ul{margin:12px 0 0;padding-left:20px;font-size:13.5px;line-height:1.6}
              li{margin-bottom:5px}
              button{margin-top:14px;width:100%;min-height:46px;border:0;
                     border-radius:10px;background:#1F7A4D;color:#fff;
                     font-size:15px;font-weight:600}
              .empty{text-align:center;padding:40px 20px;color:#4a5b51}
              .empty .glyph{font-size:52px}
              .empty h2{margin:12px 0 8px;color:#1f7a4d;font-size:19px}
              .empty p{line-height:1.6;font-size:14px;max-width:22rem;margin:0 auto}
              footer{font-size:12px;color:#7b8b82;line-height:1.5;
                     margin-top:18px;text-align:center}
            </style></head>
            <body>
              <header>
                <h1>🌱 GREENROOT</h1>
                <p>No connection — showing the ${cards.size} card${
                    if (cards.size == 1) "" else "s"} saved on this phone.
                   Pull down to try again.</p>
              </header>
              $body
              <footer>This is advice to help you decide. It is not a promise.
                Please check with your local agriculture officer before sowing.</footer>
              <script>
                // The id crosses one escaping boundary (HTML attribute) and is
                // read back as a string, never parsed as code.
                document.querySelectorAll('button.share').forEach(function (b) {
                  b.addEventListener('click', function () {
                    $BRIDGE_NAME.shareCard(b.dataset.id);
                  });
                });
              </script>
            </body></html>
            """.trimIndent(),
            "text/html", "utf-8", null,
        )
    }

    /** The card fields are user data and land inside HTML and a JS string. */
    private fun esc(s: String): String = s
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\"", "&quot;").replace("'", "&#39;")

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        web.saveState(outState)
    }
}
