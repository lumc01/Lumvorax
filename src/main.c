} else if (strcmp(argv[i], "--blackbox-universal-test") == 0) {
                printf("=== BLACKBOX UNIVERSAL MODULE TEST ===\n");
                // Test module boîte noire universel
                blackbox_config_t* config = blackbox_config_create_default();
                bool test_success = blackbox_stress_test_universal(config);
                blackbox_config_destroy(&config);

                if (test_success) {
                    printf("✅ BLACKBOX UNIVERSAL TEST: SUCCESS\n");
                    printf("Module ready for LUM/VORAX integration\n");
                } else {
                    printf("❌ BLACKBOX UNIVERSAL TEST: FAILED\n");
                }
                return test_success ? 0 : 1;
            } else if (strcmp(argv[i], "--blackbox-stealth-test") == 0) {
                printf("=== BLACKBOX STEALTH MODE TEST ===\n");
                printf("🎯 Testing maximum opacity with speed optimization\n");

                // Test mode furtivité maximale
                blackbox_config_t* stealth_config = blackbox_config_create_stealth_mode();
                bool stealth_success = blackbox_stress_test_stealth_mode(stealth_config);
                blackbox_config_destroy(&stealth_config);

                if (stealth_success) {
                    printf("🔐 BLACKBOX STEALTH TEST: SUCCESS\n");
                    printf("🚀 Maximum secrecy achieved with optimal speed\n");
                    printf("🎭 LUM/VORAX completely hidden in ML simulation\n");
                } else {
                    printf("❌ BLACKBOX STEALTH TEST: FAILED\n");
                }
                return stealth_success ? 0 : 1;