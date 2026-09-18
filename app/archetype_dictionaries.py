def get_archetype_description(sex, cluster_id, advanced_data):
    # Define archetype descriptions for mixed, men, and women categories
    archetype_descriptions_mixed_advanced = {
                    0: "a highly efficient player with strong defensive skills.",
                    1: "a strong defensive player with high defensive efficiency but prone to errors.",
                    2: "an offensive efficient player with limited usage.",
                    3: "an offensive efficient player with high usage.",
                    4: "an offensive oriented player prone to errors.",
                    5: "a master of the game with a balanced offensive and defensive skill set."
    }
    archetype_descriptions_mixed = {
                    0: "a strong defensive player with decent offensive skills.",
                    1: "an offensive oriented player prone to errors.",
                    2: "a really efficient player oriented to playmaking.",
                    3: "a strong defensive player with high efficiency but prone to errors.",
                    4: "an offensive efficient player with high usage."
    }

    archetype_descriptions_men_advanced = {
                    0: "an efficient offensive player with limited usage.",
                    1: "a player prone to errors with playmaking skills.",
                    2: "a strong defensive player with high offensive efficiency.",
                    3: "a master of the game with a balanced offensive and defensive skill set.",
                    4: "an efficient offensive player prone to errors with high usage."
    }
    archetype_descriptions_men = {
                    0: "an efficient offensive player prone to errors.",
                    1: "a strong defensive player oriented to defense and playmaking.",
                    2: "an inefficient offensive player with limited usage.",
                    3: "a strong defensive player with high offensive efficiency.",
                    4: "a master of the game with a balanced offensive and defensive skill set.",
                    5: "an efficient offensive player slightly prone to errors."
    }

    archetype_descriptions_women_advanced = {
                    0: "a defensive specialist with strong catching skills.",
                    1: "a master of the game with a balanced offensive and defensive skill set.",
                    2: "an offensive efficient player with high usage.",
                    3: "a strong defensive player with high efficiency but prone to errors."
    }
    archetype_descriptions_women = {
                    0: "a strong defensive player with high efficiency but prone to errors.",
                    1: "a defensive specialist with strong catching skills.",
                    2: "a high usage offensive player with high efficiency but slightly prone to errors.",
                    3: "a master of the game with a balanced offensive and defensive skill set.",
                    4: "an efficient offensive player but prone to errors."
    }

    # Return the appropriate description based on sex and advanced_data
    if advanced_data:
        if sex == -1:
            return archetype_descriptions_mixed_advanced.get(cluster_id, "No description available.")
        elif sex == 0:
            return archetype_descriptions_men_advanced.get(cluster_id, "No description available.")
        elif sex == 1:
            return archetype_descriptions_women_advanced.get(cluster_id, "No description available.")
    else:
        if sex == -1:
            return archetype_descriptions_mixed.get(cluster_id, "No description available.")
        elif sex == 0:      
            return archetype_descriptions_men.get(cluster_id, "No description available.")
        elif sex == 1:
            return archetype_descriptions_women.get(cluster_id, "No description available.")