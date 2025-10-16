# Update 2024-10-15

## UI Enhancements
- Added contextual ❔ hover tooltips to every data ingestion widget (tabular, image, sensor, and multimodal assets) so users know which files to prepare and which algorithms can consume them without disruptive popover clicks.
- Clarified throughout the wizards that only TEXGISA offers end-to-end multimodal optimisation, while other algorithms rely on pre-fused tables.
- Improved guidance for manifest ID fields and multimodal upload panels to reduce ambiguity and highlight required identifiers.
- Streamlined the TEXGISA expert section: λ_texgi_smooth slider removed to match the paper loss, introduced an "Important features" selector for the Ω_expert set, and refreshed helper copy to steer users through the revised interaction.

## Algorithm Routing
- Surfaced runtime feedback that automatically detects raw multimodal inputs, routes them to TEXGISA, and warns when other algorithms fall back to flat tables.
- Ensured the configuration handed to TEXGISA consistently includes multimodal sources while other algorithms ignore raw assets to avoid mismatched expectations.
- Synced the front-end regularizer controls with the updated Ω_smooth/Ω_expert implementation so that the trainer receives only the supported coefficients and the expert feature set I.

## Documentation
- Documented these adjustments for internal tracking in this `update1015.md` file.
