# Vendored third-party schemas

## `atg_filebetting_1_8_4.xsd`

ATG's file-betting ("Filinlämning") schema, fetched from
<https://www.atg.se/services/schemas/filebet/1.8.4/atg_filebetting.xsd>.
Files built by `core/reduce_system.py` are uploaded at
<https://www.atg.se/spel/reducerat>.

1.8.4 is the newest version ATG publishes, but the validator behind the
upload is **1.8.6**, which is not published anywhere. 1.8.6 adds:

- `v85Coupon` / `v85CouponType`, for the V85 pool that replaced V75 in
  October 2025. Structurally identical to `v86CouponType`: eight `<leg>`
  children plus `couponid`, `date`, `trackcode` and `betmultiplier`.
- a required coupon-level `trackcode` on `v85Coupon`, so several V85 rounds
  can run on the same day.

That element set is taken from the generated C# bindings in the open-source
HPT client (`atg_filebetting_1_8_6.cs` in
<https://github.com/Hospodaren/HPTClient>), which submits V85 files to ATG
successfully. The `schemaversion` attribute is a fixed string and stayed at
`ATG File Betting XSD ver 1.8` across both versions.

So: a V85 file cannot be validated against the vendored XSD — validate it as
a `v86Coupon` instead, which is the same shape.
`python3 -m scripts.check_reduce_system` does exactly that.
