package `in`.greenroot.app

import android.content.Context
import android.content.Intent
import android.webkit.JavascriptInterface
import android.widget.Toast
import org.json.JSONObject

/**
 * The seam between the web dashboard and the phone.
 *
 * `addJavascriptInterface` exposes these methods to whatever page the WebView
 * has loaded, so the containment is at the navigation layer instead:
 * [MainActivity] refuses to load any host but the GREENROOT origin and hands
 * external links to the system browser. Nothing here should be given a
 * capability that would matter if that guard ever failed -- note that none of
 * these methods reads arbitrary files or takes a URL.
 */
class NativeBridge(
    private val context: Context,
    private val store: CardStore,
    private val onSaved: () -> Unit,
) {

    /**
     * Called by the dashboard when the farmer saves a recommendation.
     * Returns true so the web side can confirm it landed on the phone.
     */
    @JavascriptInterface
    fun saveCard(json: String): Boolean = try {
        val o = JSONObject(json)
        val advice = o.optJSONArray("advice")
        store.save(
            SavedCard(
                id = o.optString("id", System.currentTimeMillis().toString()),
                crop = o.optString("crop"),
                confidence = o.optDouble("confidence", 0.0),
                district = o.optString("district"),
                savedAtMillis = System.currentTimeMillis(),
                readings = o.optString("readings"),
                advice = if (advice == null) emptyList()
                         else (0 until advice.length()).map { advice.optString(it) },
            )
        )
        onSaved()
        true
    } catch (e: Exception) {
        false
    }

    /** How many cards are readable with no connection. */
    @JavascriptInterface
    fun savedCount(): Int = store.all().size

    /** Hands a card to the Android share sheet — WhatsApp, SMS, print, mail. */
    @JavascriptInterface
    fun shareCard(id: String) {
        val card = store.find(id) ?: return
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "GREENROOT advice — ${card.crop}")
            putExtra(Intent.EXTRA_TEXT, card.asShareText())
        }
        context.startActivity(
            Intent.createChooser(send, "Share this advice")
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        )
    }

    /** Lets the page tell the farmer something in the platform's own idiom. */
    @JavascriptInterface
    fun toast(message: String) {
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }

    /** Present so the web app can detect it is running inside the app. */
    @JavascriptInterface
    fun isNativeApp(): Boolean = true
}
