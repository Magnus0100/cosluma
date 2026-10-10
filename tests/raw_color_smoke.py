"""Small real-ICC and EXIF decode check; no network or model inference."""
from io import BytesIO
from pathlib import Path
import tempfile
import struct

from PIL import Image, ImageCms, ImageOps

from cache_store import ContentCache
from raw_support import RGB_DECODE_VERSION, open_rgb, storage_info


def linear_srgb_profile():
    """Use standard sRGB primaries with a linear transfer curve, generated locally."""
    data = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
    count = struct.unpack_from('>I', data, 128)[0]
    for index in range(count):
        entry = 132 + index * 12
        signature, offset, size = struct.unpack_from('>4sII', data, entry)
        if signature in (b'rTRC', b'gTRC', b'bTRC'):
            assert data[offset:offset+4] == b'para' and size >= 16
            struct.pack_into('>I', data, entry+8, 16)
            struct.pack_into('>HHI', data, offset+8, 0, 0, 65536)  # ICC fixed gamma = 1.
    data[84:100] = bytes(16)  # Clear the ID after changing profile content.
    return ImageCms.ImageCmsProfile(BytesIO(data))


def main():
    profile = linear_srgb_profile()
    with tempfile.TemporaryDirectory(prefix="cosplay-icc-") as directory:
        root = Path(directory)
        tagged, untagged = root / "tagged.png", root / "untagged.png"
        pixels = Image.new("RGB", (3, 2))
        pixels.putdata([(80, 180, 100), (200, 80, 40), (50, 60, 150)] * 2)
        exif = Image.Exif(); exif[274] = 6
        pixels.save(tagged, icc_profile=profile.tobytes(), exif=exif)
        pixels.save(untagged, exif=exif)
        with Image.open(tagged) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
        result = open_rgb(tagged)
        expected = ImageCms.profileToProfile(source, profile, ImageCms.createProfile("sRGB"), outputMode="RGB")
        assert result.size == (2, 3) and result.tobytes() == expected.tobytes()
        assert result.tobytes() != source.tobytes()
        assert result.info["color_management"] == "embedded_icc_to_sRGB"
        assert "sRGB" in ImageCms.getProfileName(ImageCms.ImageCmsProfile(BytesIO(result.info["icc_profile"])))
        assumed = open_rgb(untagged)
        assert assumed.tobytes() == source.tobytes() and assumed.info["color_management"] == "assumed_sRGB"
        assert storage_info(untagged)["decode_color_management"] == "assumed_sRGB"
        cache = ContentCache(root)
        old = cache.make_key(tagged, "proxy", params={"max_side": 768}, dimensions=[2, 3])
        new = cache.make_key(tagged, "proxy", params={"max_side": 768, "decode_version": RGB_DECODE_VERSION}, dimensions=[2, 3])
        assert old != new
        print({"ok": True, "actual_profile": ImageCms.getProfileName(profile).strip(),
               "source_pixel": source.getpixel((0, 0)), "srgb_pixel": result.getpixel((0, 0)),
               "rotation_size": result.size, "untagged": "assumed_sRGB", "old_cache_not_reused": True})


if __name__ == "__main__":
    main()
