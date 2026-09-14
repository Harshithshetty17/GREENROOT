package `in`.greenroot.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * One saved recommendation, held on the phone.
 *
 * The server is authoritative while there is a connection, but a farmer
 * standing in a field with no signal still needs to read back what they were
 * told last week — which fertiliser, how much, and for which crop. That is
 * the reason this exists on-device rather than only in the server database.
 */
data class SavedCard(
    val id: String,
    val crop: String,
    val confidence: Double,
    val district: String,
    val savedAtMillis: Long,
    val readings: String,
    val advice: List<String>,
) {
    fun savedOn(): String =
        SimpleDateFormat("d MMM yyyy", Locale.getDefault()).format(Date(savedAtMillis))

    /** Plain text for the Android share sheet — WhatsApp, SMS, print. */
    fun asShareText(): String = buildString {
        append("GREENROOT — crop advice\n\n")
        append("Best crop: ").append(crop).append('\n')
        append("Match: ").append(confidence.toInt()).append(" out of 100\n")
        if (district.isNotBlank()) append("Place: ").append(district).append('\n')
        append("Soil: ").append(readings).append('\n')
        append("Saved: ").append(savedOn()).append("\n\n")
        if (advice.isNotEmpty()) {
            append("What to do:\n")
            advice.forEach { append("• ").append(it).append('\n') }
            append('\n')
        }
        append("This is advice to help you decide. It is not a promise. ")
        append("Please check with your local agriculture officer before sowing.")
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("crop", crop)
        put("confidence", confidence)
        put("district", district)
        put("savedAtMillis", savedAtMillis)
        put("readings", readings)
        put("advice", JSONArray(advice))
    }

    companion object {
        fun fromJson(o: JSONObject): SavedCard {
            val adviceArray = o.optJSONArray("advice") ?: JSONArray()
            return SavedCard(
                id = o.optString("id"),
                crop = o.optString("crop"),
                confidence = o.optDouble("confidence", 0.0),
                district = o.optString("district"),
                savedAtMillis = o.optLong("savedAtMillis", System.currentTimeMillis()),
                readings = o.optString("readings"),
                advice = (0 until adviceArray.length()).map { adviceArray.optString(it) },
            )
        }
    }
}

/**
 * A small append-only store in app-private storage.
 *
 * Deliberately a JSON file rather than Room: there is one collection, it is
 * capped, and it is read in full every time it is shown. A database would be
 * more machinery than the problem has.
 */
class CardStore(context: Context) {

    private val file = java.io.File(context.filesDir, FILE_NAME)

    fun all(): List<SavedCard> {
        if (!file.exists()) return emptyList()
        return try {
            val array = JSONArray(file.readText())
            (0 until array.length())
                .mapNotNull { array.optJSONObject(it) }
                .map { SavedCard.fromJson(it) }
                .sortedByDescending { it.savedAtMillis }
        } catch (e: Exception) {
            // A truncated write (battery pull mid-save) must not brick the
            // app; an unreadable store reads as empty and the next save
            // rewrites it.
            emptyList()
        }
    }

    /** Newest first, capped. Re-saving the same id replaces it. */
    fun save(card: SavedCard) {
        val merged = (listOf(card) + all().filter { it.id != card.id }).take(MAX_CARDS)
        val array = JSONArray().apply { merged.forEach { put(it.toJson()) } }
        val tmp = java.io.File(file.parentFile, "$FILE_NAME.tmp")
        tmp.writeText(array.toString())
        tmp.renameTo(file)   // atomic swap; a crash leaves the old file intact
    }

    fun find(id: String): SavedCard? = all().firstOrNull { it.id == id }

    fun delete(id: String) {
        val kept = all().filter { it.id != id }
        val array = JSONArray().apply { kept.forEach { put(it.toJson()) } }
        file.writeText(array.toString())
    }

    private companion object {
        const val FILE_NAME = "saved_cards.json"
        const val MAX_CARDS = 100
    }
}
