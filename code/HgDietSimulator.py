import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import os, sys
from pathlib import Path


#color mapping for protein substitution and degree of global warming
protein_colors = {
    'Interior with Salmon':'#1f77b4',
     'Interior without Salmon (non-salmon)':'#2ca02c',
    'Interior without Salmon (moose)':'#ff7f0e',

    'Coastal with Salmon':'#1f77b4',
     'Coastal without Salmon (non-salmon)':'#2ca02c',
    'Coastal without Salmon (moose)':'#ff7f0e'
    }
condition_colors = {
    'Normal':'#1f77b4',
    'Moderate':'#E9D028',
    'Severe':'#A10004',
    }

# import data file

code_dir = Path(os.getcwd())
data_dir = code_dir.parent/'data'
data = pd.read_csv(data_dir/'Diet Model Data.csv')


WCal = 2020 # female daily calorie intake
WKg = 77.11  # average woman's weight (kg)

''' # Define required percentages and number of resources for interior and coastal regions
    # naming convention 
        # location[suffix]_required_percentages
            # wosns-w/o salmon non-salmon replacement
            # wosm-w/o salmon moose/caribou replacement
        # location[suffix_selection_counts 
            # wos-w/o salmon'''

coastal_required_percentages = { # diet survey
    # Traditional Foods
    "Animal Products":0.0003,
    "Non-Salmon":0.0707,
    "Game": 0.0262,
    "Moose and Caribou": 0.0843,
    "Salmon": 0.1262,
    "Marine Mammal": 0.0279,


    # Western Foods
    "Sugared Drinks": 0.16,
    "Store Fruit and Fruit Juice": 0.07,
    "Dairy and Eggs": 0.08,
    "Grains": 0.12,
    "Imported Protein": 0.07,
    "Vegetable Fats": 0.07,
    "Mixed Dishes": 0.05,
    "Sweeteners and Candy": 0.03
}
coastal_selection_counts = {
    # Traditional Foods
    "Animal Products": 1,
    "Non-Salmon": 1,
    "Game": 1,
    "Moose and Caribou": 1,
    "Salmon": 1,
    "Marine Mammal": 1,


    # Western Foods
    "Sugared Drinks": 1,
    "Store Fruit and Fruit Juice": 2,
    "Dairy and Eggs": 1,
    "Grains": 1,
    "Imported Protein": 1,
    "Vegetable Fats": 1,
    "Mixed Dishes": 1,
    "Sweeteners and Candy": 1

}
coastalwosns_required_percentages = {
    #replaced salmon with non-salmon
    # Traditional Foods
    "Animal Products":0.0003,
    "Non-Salmon":0.1969,
    "Game": 0.0262,
    "Moose and Caribou": 0.0843,
    #"Salmon": 0.1262,
    "Marine Mammal": 0.0279,
    #"Traditional Plants and Fungi":0,

    # Western Foods
    "Sugared Drinks": 0.16,
    "Store Fruit and Fruit Juice": 0.07,
    "Dairy and Eggs": 0.08,
    "Grains": 0.12,
    "Imported Protein": 0.07,
    "Vegetable Fats": 0.07,
    "Mixed Dishes": 0.05,
    "Sweeteners and Candy": 0.03
}
coastalwosm_required_percentages = {
    # Traditional Foods
    "Animal Products": 0.0003,
    "Non-Salmon": 0.0707,
    "Game": 0.0262,
    "Moose and Caribou": 0.2105,
    #"Salmon": 0.1262,
    "Marine Mammal": 0.0279,

    # Western Foods
    "Sugared Drinks": 0.16,
    "Store Fruit and Fruit Juice": 0.07,
    "Dairy and Eggs": 0.08,
    "Grains": 0.12,
    "Imported Protein": 0.07,
    "Vegetable Fats": 0.07,
    "Mixed Dishes": 0.05,
    "Sweeteners and Candy": 0.03
}
coastalwos_selection_counts = {
    # Traditional Foods
    "Animal Products": 1,
    "Non-Salmon": 1,
    "Game": 1,
    "Moose and Caribou": 1,
    #"Salmon":0,
    "Marine Mammal": 1,

    # Western Foods
    "Sugared Drinks": 1,
    "Store Fruit and Fruit Juice": 2,
    "Dairy and Eggs": 1,
    "Grains": 1,
    "Imported Protein": 1,
    "Vegetable Fats": 1,
    "Mixed Dishes": 1,
    "Sweeteners and Candy": 1
}

interior_required_percentages = {
    # Traditional Foods
    "Non-Salmon": 0.0191,
    "Game": 0.0147,
    "Moose and Caribou": 0.0635,
    "Salmon": 0.0809,
    "Traditional Plants and Fungi": 0.0018,

    # Western Foods
    "Sugared Drinks": 0.12,
    "Store Vegetable": 0.05,
    "Store Fruit and Fruit Juice": 0.11,
    "Dairy and Eggs": 0.13,
    "Grains": 0.15,
    "Imported Protein": 0.07,
    "Vegetable Fats": 0.1,
    "Mixed Dishes": 0.06,
}
interior_selection_counts = {
    # Traditional Foods
    "Non-Salmon": 1,
    "Game": 1,
    "Moose and Caribou": 1,
    "Salmon": 1,
    "Traditional Plants and Fungi": 1,

    # Western Foods
    "Sugared Drinks": 1,
    "Store Vegetable": 3,
    "Store Fruit and Fruit Juice": 2,
    "Dairy and Eggs": 1,
    "Grains": 1,
    "Imported Protein": 1,
    "Vegetable Fats": 1,
    "Mixed Dishes": 1,
}
interiorwosns_required_percentages = {
    # Traditional Foods
    "Non-Salmon": .1000,
    "Game": 0.0147,
    "Moose and Caribou": 0.0635,
    #"Salmon": 0.0809,
    "Traditional Plants and Fungi": 0.0018,

    # Western Foods
    "Sugared Drinks": 0.12,
    "Store Vegetable": 0.05,
    "Store Fruit and Fruit Juice": 0.11,
    "Dairy and Eggs": 0.13,
    "Grains": 0.15,
    "Imported Protein": 0.07,
    "Vegetable Fats": 0.1,
    "Mixed Dishes": 0.06,
}
interiorwosm_required_percentages = {
    # Traditional Foods
    "Non-Salmon": 0.0191,
    "Game": 0.0147,
    "Moose and Caribou": 0.1444,
    #"Salmon": 0.0809,
    "Traditional Plants and Fungi": 0.0018,

    # Western Foods
    "Sugared Drinks": 0.12,
    "Store Vegetable": 0.05,
    "Store Fruit and Fruit Juice": 0.11,
    "Dairy and Eggs": 0.13,
    "Grains": 0.15,
    "Imported Protein": 0.07,
    "Vegetable Fats": 0.1,
    "Mixed Dishes": 0.06,
}
interiorwos_selection_counts = { # this replaces salmon with moose
    # Traditional Foods
    "Non-Salmon": 1,
    "Game": 1,
    "Moose and Caribou": 1,
    #"Salmon": 0,
    "Traditional Plants and Fungi": 1,

    # Western Foods
    "Sugared Drinks": 1,
    "Store Vegetable": 3,
    "Store Fruit and Fruit Juice": 2,
    "Dairy and Eggs": 1,
    "Grains": 1,
    "Imported Protein": 1,
    "Vegetable Fats": 1,
    "Mixed Dishes": 1,
}


def filter_food_by_area(df, protein_type):
    # this function selects what foods the program can choose from based on region
    if protein_type == "Interior with Salmon":
        filtered_df = df[df["Coastal/Interior"].isin(['Imported', 'Local','Interior'])]
    elif protein_type == "Interior without Salmon (non-salmon)":
        filtered_df = df[df["Coastal/Interior"].isin(['Imported', 'Local','Interior'])]
    elif protein_type == "Interior without Salmon (moose)":
        filtered_df = df[df["Coastal/Interior"].isin(['Imported', 'Local','Interior'])]
    elif protein_type == "Coastal with Salmon":
        filtered_df = df[df["Coastal/Interior"].isin(['Imported', 'Local','Coastal'])]
    elif protein_type == "Coastal without Salmon (non-salmon)":
        filtered_df = df[df["Coastal/Interior"].isin(['Imported', 'Local','Coastal'])]
    elif protein_type == "Coastal without Salmon (moose)":
        filtered_df = df[df["Coastal/Interior"].isin(['Imported', 'Local','Coastal'])]
    else:
        print("Invalid Choice") # No filtering if protein_type is "All"

    return filtered_df

def create_diet(data, calorie_cap, protein_type, condition,threshold=0.05, max_iterations=1000):
    '''
    # This function creates 1 random diet based on set parameters and returns the foods that make up the diet. Returns
    required calories, total mercury content, total calories, and the calories attributed to traditional foods
        # data - variable holding model data
        # calorie_cap - max number of calories eaten a day
        # protein_type- [location] with/without salmon (protein substituted)
        # condition - degree of warming [normal, moderate,severe]
        # threshold - how close to calorie cap created diet is
        # max_iterations- number of times the program tries to find a diet before it quits'''


    if condition == "Normal":
        Hg ='THg (ng/g)'
    elif condition == "Moderate":
        Hg ='Moderate Hg Increase'
    elif condition == "Severe":
        Hg ='Severe Hg Increase'
    else:
        Hg ='THg (ng/g)'

    if protein_type == "Interior with Salmon":
        required_percentages = {group: round(percentage, 4) for group, percentage in interior_required_percentages.items()}
        required_count = interior_selection_counts
    elif protein_type == "Interior without Salmon (non-salmon)":
        required_percentages = {group: round(percentage, 4) for group, percentage in interiorwosns_required_percentages.items()}
        required_count = interiorwos_selection_counts
    elif protein_type == "Interior without Salmon (moose)":
        required_percentages = {group: round(percentage, 4) for group, percentage in interiorwosm_required_percentages.items()}
        required_count = interiorwos_selection_counts
    elif protein_type == "Coastal with Salmon":
        required_percentages = {group: round(percentage, 4) for group, percentage in coastal_required_percentages.items()}
        required_count = coastal_selection_counts
    elif protein_type == "Coastal without Salmon (moose)":
        required_percentages = {group: round(percentage, 4) for group, percentage in coastalwosm_required_percentages.items()}
        required_count = coastalwos_selection_counts
    elif protein_type == "Coastal without Salmon (non-salmon)":
        required_percentages = {group: round(percentage, 4) for group, percentage in coastalwosns_required_percentages.items()}
        required_count = coastalwos_selection_counts
    else:
        raise ValueError('Please enter a valid option')

    # Calculate the sum of the percentages for the modeled portion of the diet (percentages were not 100%)
    modeled_percentage = round(sum(required_percentages.values()), 2)  # Fraction of total diet being modeled

    # Adjust the calorie cap based on the modeled portion
    adjusted_calorie_cap = round(calorie_cap * modeled_percentage, 2)

    # Filter foods by area based on protein type
    filtered_data = filter_food_by_area(data, protein_type)

    # Initialize variables
    selected_foods = []
    current_calories = 0
    total_traditional_calories = 0
    total_mercury = 0
    iteration = 0

    while iteration < max_iterations:
        selected_foods.clear()  # Reset selected foods for each attempt
        current_calories = 0
        total_traditional_calories = 0
        total_mercury = 0

        # Loop over each food group to select items based on count and scale by calories
        for group, count in required_count.items():
            group_items = filtered_data[filtered_data['Food Source'] == group]

            if not group_items.empty:
                # Randomly sample the specified number of items for the food group
                selected_items = group_items.sample(n=count, replace=True).copy()

                # Calculate the target calories for this group
                required_calories = round(calorie_cap* required_percentages[group], 2)

                per_item_calories = required_calories / count  # Split group target calories across items

                # Set 'Calories per 100g' as a direct reference to the item's calorie density
                selected_items['Calories'] = selected_items['Calories']

                # Calculate the scaled amount and scaled calories for each item
                selected_items['Scaled Amount (g)'] = round(
                    (per_item_calories * selected_items['Amount (g)']) / selected_items['Calories'], 2)
                selected_items['Scaled Calories'] = round((selected_items['Scaled Amount (g)'] * selected_items[
                    'Calories']) / 100,3)

                # Calculate traditional calories and mercury content
                selected_items['Traditional Calories'] = selected_items['Scaled Calories'] * selected_items[
                    'Traditional Food']
                selected_items['Mercury Content'] = round((selected_items[Hg] / 1000) * selected_items[
                    'Scaled Amount (g)'],3 ) # in μg

                # Sum values for each food group
                total_traditional_calories += selected_items['Traditional Calories'].sum()
                total_mercury += selected_items['Mercury Content'].sum()
                selected_foods.extend(selected_items.to_dict(orient='records'))
                current_calories += selected_items['Scaled Calories'].sum()

        # Check if total calories are within threshold based on the adjusted calorie cap
        if abs(current_calories - adjusted_calorie_cap) / adjusted_calorie_cap <= threshold:
            break

        iteration += 1

    if iteration == max_iterations:
        print("Warning: Could not find a diet meeting the calorie cap within the specified threshold.")
        return pd.DataFrame(), 0, 0, 0  # Return empty DataFrame and zero totals if no diet found

    # Combine selected items into a DataFrame
    diet_df = pd.DataFrame(selected_foods)
    required_calories_dict = {group: round(calorie_cap * required_percentages[group], 2) for group in
                              required_percentages}
    return diet_df, round(current_calories, 2), round(total_traditional_calories, 2), round(total_mercury, 2),required_calories_dict

def generate_multiple_diets(data, calorie_cap, protein_type, conditions, threshold=0.05, num_diets=10):
    ''''
      this function runs the create_diet function multiple time to create (x=num_diets) number of diets
      and returns information on the diets generated
        # data - variable holding model data
        # calorie_cap - max number of calories eaten a day
        # protein_type- [location] with/without salmon (protein substituted)
        # conditions - degree of warming [normal, moderate,severe] needs to be in brackets
        # threshold - how close to calorie cap created diet is
        # num_diets- number of diets generated  '''

    diets = []

    for condition in conditions:
        for i in range(num_diets):  # Run the diet generation for the current condition
            #print(f"Generating {protein_type} diet {i + 1} for {condition} conditions ...")  # Debug print
            diet_df, total_calories, traditional_calories, mercury_content, target_calories = create_diet(
                data, calorie_cap, protein_type, condition, threshold
            )

            # Check if diet generation was successful (i.e., non-empty DataFrame returned)
            if not diet_df.empty:
                diets.append({
                    'Diet': diet_df,
                    'Total Calories': total_calories,
                    'Target Calories': target_calories,
                    'Traditional Calories': traditional_calories,
                    'Mercury Content': mercury_content
                })

            else:
                print(f"Diet {i + 1} for condition '{condition}' generation failed to meet calorie requirements.")

    return diets

def permafrost_thaw_distribution(protein_type, conditions, cal_cap, num_runs=1000, tolerance=0.05): # use this to compare warming conditions
    """ Generates and plots a distribution of total THg from diets under different thawing permafrost conditions. """
    # Store the THg data for each condition
    thg_data = {condition: [] for condition in conditions}
    diet_data = {condition: [] for condition in conditions}

    # Iterate over each condition
    for condition in conditions:
        #print(f"Generating diets for condition: {condition}")

        # Generate multiple diets using generate_multiple_diets
        diets = generate_multiple_diets(
            data,
            calorie_cap=cal_cap,
            protein_type=protein_type,
            threshold=tolerance,
            num_diets=num_runs,
            conditions=[condition]
        )

        # Extract total THg values from each generated diet
        for diet in diets:
            thg_data[condition].append(diet['Mercury Content'])
            diet_data[condition].append(diet['Diet'])


    # Plot the distribution of total THg for each warming condition
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through the conditions and plot the data
    for condition in conditions:
        sns.kdeplot(thg_data[condition], fill=True, label=condition, ax=ax,color=condition_colors.get(condition, "gray"))

    # Customize the plot
    ax.set_xlabel('Mercury Intake (THg µg/day)',fontsize=14)
    ax.set_ylabel('Density',fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlim(0, 500)
    ax.set_title(f'Distribution of Total THg for {num_runs} Diets (Permafrost Thaw)')
    ax.legend(title='Condition')

    plt.tight_layout()
    plt.show()

    return thg_data, diet_data

def protein_distribution(protein_types, cal_cap, conditions, num_runs=1000, tolerance=0.05): #use this to plot different protein types
    ''' Generates and plots distribution of total THg from diet that substitute salmon for alternative proteins'''

    thg_data = {protein: [] for protein in protein_types}
    diet_data = {protein: [] for protein in protein_types}

    # Loop over each protein type to generate diets
    for protein_type in protein_types:

        diets = generate_multiple_diets(
            data,
            calorie_cap=cal_cap,
            protein_type=protein_type,
            conditions=conditions,
            num_diets=num_runs,
            threshold=tolerance
        )

        # Collect the mercury content (THg) for each diet generated
        for diet in diets:
            thg_data[protein_type].append(diet['Mercury Content'])
            diet_data[protein_type].append(diet['Diet'])


    # Create the plot to compare distributions
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Loop through each protein type and plot its distribution
    for protein_type in protein_types:
        sns.kdeplot(thg_data[protein_type], fill=True, label=protein_type, ax=ax1, color=protein_colors.get(protein_type, "gray"))

    # Customize the plot
    ax1.set_xlim(0, 300)
    ax1.set_xlabel('Mercury Intake (THg µg/day)',fontsize=14)
    ax1.set_ylabel('Density',fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_title(f'Distribution of Total THg for {num_runs} Diets (Protein Substitution)')
    ax1.legend(title='Protein Type')

    plt.tight_layout()
    plt.show()

    # Return THg data and diet data
    return thg_data, diet_data

def extract_new_targets(diet_data, thg_data, new_targets, tolerance=3.0, top_n=5):
    ''' This function finds diets with the THg closest to the requested amount
        # diet_data- hold list of items in 1 diet
        # thg_data- holds thg for the associated diet
        # new_targets- target THg values
        # tolerance - how close to the requested value returned values must be
        # top_n- how many values are returned

        '''
    if not isinstance(new_targets, list):
        new_targets = [new_targets]

    # Initialize the result dictionary
    closest_diets = {target: {protein: [] for protein in diet_data} for target in new_targets}

    # Loop through each target THg value
    for target in new_targets:
        for protein_type in diet_data:
            # Filter diets within the tolerance range of the current target THg
            matching_diets = [
                {"Diet": diet, "THg": thg}
                for diet, thg in zip(diet_data[protein_type], thg_data[protein_type])
                if abs(thg - target) <= tolerance
            ]

            # Sort diets by THg (ascending order) and select top_n
            sorted_diets = sorted(matching_diets, key=lambda x: abs(x['THg'] - target))[:top_n]

            closest_diets[target][protein_type] = sorted_diets

    return closest_diets

def proteinscenario(store_file_name='protein_type_simulation_results.pkl', num_diets=100, region="Coastal"):
    ''' # Runs simulations for all protein substitution scenarios
    # store_file_name- name of file that is holding simulation data
    # region - what part of the Yukon River Basin [Interior or Coastal]
    '''

    num_diets = num_diets  # Define the number of diets you want to generate
    calorie_cap = WCal  # enter desired calorie cap

    # Compares switching to from salmon to other protein alternatives
    protein_condition = ['Normal']
    if region == 'Coastal':
        protein_type = ['Coastal with Salmon', 'Coastal without Salmon (moose)', 'Coastal without Salmon (non-salmon)']
    elif region == 'Interior':
        protein_type = ['Interior with Salmon','Interior without Salmon (moose)','Interior without Salmon (non-salmon)']
    else:
        raise ValueError('Please enter a valid option ("Coastal","Interior")')

    thg_data, diet_data = protein_distribution(protein_types=protein_type, conditions=protein_condition,
                                               cal_cap=calorie_cap, num_runs=num_diets, tolerance=0.01);
    # Save results for reuse
    pd.to_pickle((thg_data, diet_data, protein_type, protein_condition), store_file_name)  # Save as a pickle file
    print(f"Simulation results saved.")
    return thg_data, diet_data

def permafrostthaw(store_file_name='permafrost_thaw_simulation_results.pkl', num_diets=10000, region='Coastal'):

    ''' # Runs simulations for all permafrost thaw scenarios
        # store_file_name- name of file that is holding simulation data
        # region - what part of the Yukon River Basin [Interior or Coastal]
        '''

    num_diets = num_diets  # Define the number of diets you want to generate
    calorie_cap = WCal # Define calorie cap

    if region == 'Coastal':
        thgprotein_type = "Coastal with Salmon"
    elif region == 'Interior':
        thgprotein_type = "Interior with Salmon"
    else:
        raise ValueError('Please enter a valid option ("Coastal","Interior")')

    thgcondition = ['Normal','Moderate', 'Severe']

    thg_data,diet_data = permafrost_thaw_distribution(protein_type=thgprotein_type, conditions=thgcondition, cal_cap=calorie_cap, num_runs=num_diets, tolerance=0.01);

    # Save results for reuse
    pd.to_pickle((thg_data, diet_data, thgprotein_type, thgcondition), store_file_name)  # Save as a pickle file
    print(f"Simulation results saved.")
    return thg_data, diet_data

def both(store_file_name='both_simulation_results.pkl', num_diets=10000,region='Coastal'):

    ''' # Runs simulations for scenarios with both moderate warming and protein substitution
        # store_file_name- name of file that is holding simulation data
        # region - what part of the Yukon River Basin [Interior or Coastal]
    '''


    num_diets = num_diets  # Define the number of diets you want to generate
    calorie_cap = WCal  # Define calorie cap

    protein_condition = ['Moderate']
    if region =='Coastal':
        protein_type = ['Coastal with Salmon', 'Coastal without Salmon (moose)', 'Coastal without Salmon (non-salmon)']
    elif region =='Interior':
        protein_type = ['Interior with Salmon', 'Interior without Salmon (moose)',
                        'Interior without Salmon (non-salmon)']
    else:
        raise ValueError('Please enter a valid option ("Coastal","Interior")')

    thg_data, diet_data = protein_distribution(protein_types=protein_type, conditions=protein_condition,
                                               cal_cap=calorie_cap, num_runs=num_diets, tolerance=0.01);
    # Save results for reuse
    pd.to_pickle((thg_data, diet_data, protein_type, protein_condition), store_file_name)  # Save as a pickle file
    print(f"Simulation results saved.")

    return thg_data, diet_data

def permafrost_thaw_yearly_distribution(file_name='simulation_results.pkl', condition=None, num_samples=365,num_runs=1000):

    ''' Randomly selects 365 diets from the 10,000 generated daily diets
    to generate yearly Hg intake under permafrost thawing conditions
    # file_name- name of file that is holding simulation data
    # condition - degree of warming
    # num_samples- number of days randomly sampled
    # num_run- how many times you sample
    '''

    if condition is None:
        print("Enter a correct value")

    # Load data from the pickle file
    loaded_data = pd.read_pickle(file_name)

    # Unpack the data into appropriate variables
    thg_data, diet_data, protein_data, condition_data = loaded_data  # Make sure variable names match

    # Store results for each condition
    results = {}
    for cond in condition:
        # Ensure condition is in thg_data
        if cond not in thg_data:
            raise ValueError(f"Condition '{cond}' not found in THg data.")

        # Extract the THg data for the selected condition
        thg_values = np.array(thg_data[cond])

        # Store the summed THg values for each run
        summed_values = []
        for _ in range(num_runs):
            random_sample = np.random.choice(thg_values, size=num_samples, replace=True)
            summed_values.append(np.sum(random_sample))

        # Calculate mean and standard deviation
        mean_thg = np.mean(summed_values)
        std_thg = np.std(summed_values)

        # Store results for this condition
        results[cond] = {
            'mean': mean_thg,
            'std_dev': std_thg,
        }

    return results

def proteinyearly_distribution(file_name='simulation_results.pkl', proteins=None, num_samples=365, num_runs=1000): #use for protein and both

    '''Randomly selects 365 diets from the 10,000 generated daily diets
    to generate yearly Hg intake under protein substitution scenarios
       # file_name- name of file that is holding simulation data
       # proteins - what proteins are being consumed
       # num_samples- number of days randomly sampled
       # num_run- how many times you sample
       '''
    if proteins is None:
        print("Enter a correct value")  # Default to a single protein if none is provided

    # Load data from the pickle file
    loaded_data = pd.read_pickle(file_name)

    # Unpack the data into appropriate variables
    thg_data, diet_data, protein_data, condition_data = loaded_data  # Make sure variable names match

    # Store results for each protein
    results = {}
    for protein in proteins:
        # Ensure protein is in thg_data
        if protein not in thg_data:
            raise ValueError(f"Protein '{protein}' not found in THg data.")

        # Extract the THg data for the selected protein
        thg_values = np.array(thg_data[protein])

        # Store the summed THg values for each run
        summed_values = []
        for _ in range(num_runs):
            random_sample = np.random.choice(thg_values, size=num_samples, replace=True)
            summed_values.append(np.sum(random_sample))

        # Calculate mean and standard deviation
        mean_thg = np.mean(summed_values)
        std_thg = np.std(summed_values)

        # Store results for this protein
        results[protein] = {
            'mean': mean_thg,
            'std_dev': std_thg,
        }
    return results

def exportpeakdata(simulation='Permafrost Thaw',loadfile="simulation_results.pkl", region='Coastal', exportname ="new_target_results.csv",targets=[10,15]):
    ''' # This function lets you define peaks from distribution plots and export diets that compose the targeted peaks
    # simulation- 'Permafrost Thaw' or 'Protein'
    loadfile- name of file containing the precomputed simulation data
    region- 'Interior' or 'Coastal'
    exportname- file name of exported peak data
    '''

    thg_data, diet_data, protein_types, conditions = pd.read_pickle(loadfile)

    if simulation == "Protein":
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        # Loop through each protein type and plot its distribution
        for protein_type in protein_types:
            sns.kdeplot(thg_data[protein_type], fill=True, label=protein_type, ax=ax1,color=protein_colors.get(protein_type, "gray"))

        # can adjust x and y axis limits if needed
        if region == "Coastal":
            ax1.set_xlim(0, 375)
            ax1.set_ylim(0, 0.04)
        elif region== "Interior":
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 0.25)
        elif region=="Moderate Interior":
            ax1.set_xlim(0, 120)
            ax1.set_ylim(0, 0.20)
        elif region=="Moderate Coastal":
            ax1.set_xlim(0, 300)
            ax1.set_ylim(0, 0.035)
        else:
            raise ValueError('Please enter a valid option ("Coastal", "Interior", "Moderate Interior", "Moderate Coastal")')

        ax1.tick_params(axis='x', labelsize=20)
        ax1.tick_params(axis='y', labelsize=20)
        ax1.set_xlabel('Mercury Intake (THg µg/day)',fontsize=20)
        ax1.set_ylabel('Density',fontsize=20)

        ax1.legend(title='Protein Substitution',title_fontsize=20,fontsize=18)

        plt.tight_layout()
        plt.show()

        # Extract new diets based on target THg values
        new_targets = targets
        results = extract_new_targets(diet_data, thg_data, new_targets, tolerance=0.5)

        # Prepare a list to store the rows for the CSV
        rows = []

        # Loop through the results and prepare the rows
        for target, protein_results in results.items():
            for protein, diets in protein_results.items():
                for diet in diets:
                    # Ensure 'Diet' is a list and handle it properly for CSV export
                    if isinstance(diet["Diet"], list):
                        diet_str = ", ".join(diet["Diet"])
                    else:
                        diet_str = str(diet["Diet"])  # Just in case it's not a list

                    # Append the row to the rows list
                    rows.append({
                        "Target THg": target,
                        "Protein Type": protein,
                        "THg": diet["THg"],
                        "Diet": diet_str
                    })
    elif simulation == "Permafrost Thaw":

        fig, ax = plt.subplots(figsize=(12,8))
        # Loop through the conditions and plot the data
        for condition in conditions:
            sns.kdeplot(thg_data[condition], fill=True, label=condition, ax=ax,color=condition_colors.get(condition, "gray"))

        # Customize the plot
        ax.set_xlabel('Mercury (THg µg/day)',fontsize=20)
        ax.set_ylabel('Density',fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        # can adjust x and y axis limits if needed
        if region =="Coastal":
            ax.set_xlim(0, 375)
            ax.set_ylim(0, 0.04)
        elif region=="Interior":
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 0.25)
        else:
            raise ValueError('Please enter a valid option ("Coastal", "Interior")')

        ax.legend(title='Permafrost Thaw',title_fontsize=20,fontsize=18)

        plt.tight_layout()
        plt.show()

        # Extract new diets based on target THg values
        new_targets = targets
        results = extract_new_targets(diet_data, thg_data, new_targets, tolerance=0.5)

        # Prepare a list to store the rows for the CSV
        rows = []

        # Loop through the results and prepare the rows
        for target, climatecondition_results in results.items():
            for climatecondition, diets in climatecondition_results.items():
                for diet in diets:
                    # Ensure 'Diet' is a list and handle it properly for CSV export
                    if isinstance(diet["Diet"], list):
                        diet_str = ", ".join(diet["Diet"])
                    else:
                        diet_str = str(diet["Diet"])  # Just in case it's not a list

                    # Append the row to the rows list
                    rows.append({
                        "Target THg": target,
                        "Climate Condition": climatecondition,
                        "THg": diet["THg"],
                        "Diet": diet_str
                    })
    else:
        print("Please choose a valid option")

    # Create a DataFrame from the rows
    df = pd.DataFrame(rows)

    # Check the DataFrame to ensure it contains the correct data
    #print(f"DataFrame before exporting:\n{df}")
    print(f"Length of exported DataFrame: {len(df)}, rows containing missing values: {df.isnull().any(axis=1).sum()}")

    # Export to CSV
    df.to_csv(exportname, index=False)
    print(f"Results saved.")

def yearly_distributions_data():
    ''' Creates Bar plots and stddev error bars for all the different simulations '''

    # Yearly Control Values
    hi = proteinyearly_distribution(file_name='interior protein 10k second draft.pkl', proteins=['Interior with Salmon'],
                                    num_samples=365, num_runs=10000)
    ciao = proteinyearly_distribution(file_name='coastal protein 10k second draft.pkl', proteins=['Coastal with Salmon'],
                                      num_samples=365, num_runs=10000)

    # Yearly Protein Scenarios
    bye = proteinyearly_distribution(file_name='interior protein 10k second draft.pkl', proteins=['Interior without Salmon (moose)'],
                                     num_samples=365, num_runs=10000)
    goodbye = proteinyearly_distribution(file_name='interior protein 10k second draft.pkl',proteins=['Interior without Salmon (non-salmon)'],
                                         num_samples=365, num_runs=10000)

    bonjour = proteinyearly_distribution(file_name='coastal protein 10k second draft.pkl', proteins=['Coastal without Salmon (moose)'],
                                         num_samples=365, num_runs=10000)

    aloha = proteinyearly_distribution(file_name='coastal protein 10k second draft.pkl',
                                       proteins=['Coastal without Salmon (non-salmon)'], num_samples=365, num_runs=10000)

    # Yearly Climate Scenarios
    nihao = permafrost_thaw_yearly_distribution(file_name='interior permafrost thaw 10k second draft.pkl',
                                             condition=['Moderate'], num_samples=365, num_runs=10000)
    annyeong = permafrost_thaw_yearly_distribution(file_name='interior permafrost thaw 10k second draft.pkl',
                                                condition=['Severe'], num_samples=365, num_runs=10000)
    bonvoyage = permafrost_thaw_yearly_distribution(file_name='coastal permafrost thaw 10k second draft.pkl',
                                                 condition=['Moderate'], num_samples=365, num_runs=10000)
    bienvenue = permafrost_thaw_yearly_distribution(file_name='coastal permafrost thaw 10k second draft.pkl',
                                                 condition=['Severe'], num_samples=365, num_runs=10000)


    # Yearly moderate permafrost thaw and protein substitution scenarios
    salut = proteinyearly_distribution(file_name='moderate interior both 10k second draft.pkl',
                                     proteins=['Interior without Salmon (moose)'],
                                     num_samples=365, num_runs=10000)
    aurevoir = proteinyearly_distribution(file_name='moderate interior both 10k second draft.pkl',
                                         proteins=['Interior without Salmon (non-salmon)'], num_samples=365,
                                         num_runs=10000)
    adios = proteinyearly_distribution(file_name='moderate coastal both 10k second draft.pkl',
                                         proteins=['Coastal without Salmon (moose)'],
                                         num_samples=365, num_runs=10000)
    gaseyo = proteinyearly_distribution(file_name='moderate coastal both 10k second draft.pkl',
                                       proteins=['Coastal without Salmon (non-salmon)'], num_samples=365,
                                       num_runs=10000)

    # Calculates Mean Values
    # Control scenarios (Coastal and Interior)
    coastal_control = ciao['Coastal with Salmon']['mean']
    interior_control = hi['Interior with Salmon']['mean']

    # Protein scenarios (without Salmon - non-salmon, moose)
    coastal_non_salmon = aloha['Coastal without Salmon (non-salmon)']['mean']
    interior_non_salmon = goodbye['Interior without Salmon (non-salmon)']['mean']

    coastal_moose = bonjour['Coastal without Salmon (moose)']['mean']
    interior_moose = bye['Interior without Salmon (moose)']['mean']

    # Climate change scenarios (Moderate and Severe)
    coastal_moderate = bonvoyage['Moderate']['mean']
    interior_moderate = nihao['Moderate']['mean']

    coastal_severe = bienvenue['Severe']['mean']
    interior_severe = annyeong['Severe']['mean']

    # both (moderate with protein change)
    moderate_coastal_non_salmon = gaseyo['Coastal without Salmon (non-salmon)']['mean']
    moderate_interior_non_salmon = aurevoir['Interior without Salmon (non-salmon)']['mean']

    moderate_coastal_moose = adios['Coastal without Salmon (moose)']['mean']
    moderate_interior_moose = salut['Interior without Salmon (moose)']['mean']

    # Calculates standard deviations
    coastal_control_std = ciao['Coastal with Salmon']['std_dev']
    interior_control_std = hi['Interior with Salmon']['std_dev']

    coastal_non_salmon_std = aloha['Coastal without Salmon (non-salmon)']['std_dev']
    interior_non_salmon_std = goodbye['Interior without Salmon (non-salmon)']['std_dev']

    coastal_moose_std = bonjour['Coastal without Salmon (moose)']['std_dev']
    interior_moose_std = bye['Interior without Salmon (moose)']['std_dev']

    # Moderate permafrost thaw with diet change
    moderate_coastal_non_salmon_std = gaseyo['Coastal without Salmon (non-salmon)']['std_dev']
    moderate_interior_non_salmon_std = aurevoir['Interior without Salmon (non-salmon)']['std_dev']

    moderate_coastal_moose_std = adios['Coastal without Salmon (moose)']['std_dev']
    moderate_interior_moose_std = salut['Interior without Salmon (moose)']['std_dev']

    coastal_moderate_std = bonvoyage['Moderate']['std_dev']
    interior_moderate_std = nihao['Moderate']['std_dev']

    coastal_severe_std = bienvenue['Severe']['std_dev']
    interior_severe_std = annyeong['Severe']['std_dev']

    # Labels and Colors
    axis_labels = ['Control', 'A', 'B', 'C',
              'D', 'A,C', 'B,C']

    coastal_means = [coastal_control, coastal_non_salmon, coastal_moose, coastal_moderate, coastal_severe, moderate_coastal_non_salmon, moderate_coastal_moose]
    interior_means = [interior_control, interior_non_salmon, interior_moose, interior_moderate, interior_severe, moderate_interior_non_salmon, moderate_interior_moose]

    coastal_std_devs = [coastal_control_std, coastal_non_salmon_std, coastal_moose_std, coastal_moderate_std,
                        coastal_severe_std,moderate_coastal_non_salmon_std, moderate_coastal_moose_std]
    interior_std_devs = [interior_control_std, interior_non_salmon_std, interior_moose_std, interior_moderate_std,
                         interior_severe_std,moderate_interior_non_salmon_std, moderate_interior_moose_std]

    # Print out the means and standard deviations for coastal and interior scenarios
    print("Coastal Means and Standard Deviations:")
    for label, mean, std_dev in zip(axis_labels, coastal_means, coastal_std_devs):
        print(f"{label}: Mean = {mean:.2f}, Standard Deviation = {std_dev:.2f}")

    print("\nInterior Means and Standard Deviations:")
    for label, mean, std_dev in zip(axis_labels, interior_means, interior_std_devs):
        print(f"{label}: Mean = {mean:.2f}, Standard Deviation = {std_dev:.2f}")


    fig, ax = plt.subplots(figsize=(10, 6))


    x_pos = np.arange(len(axis_labels))
    x_pos_interior = x_pos + 0.35

    bars_coastal = ax.bar(x_pos, coastal_means, 0.35, facecolor='cornflowerblue', edgecolor='black',
                          yerr=coastal_std_devs, capsize=5, label='Coastal')

    bars_interior = ax.bar(x_pos_interior, interior_means, 0.35, facecolor='seagreen', edgecolor='black',
                           yerr=interior_std_devs, capsize=5, label='Interior')


    # Add a red line at y=21310 for total yearly mercury intake (µg)
    ax.axhline(y=22484, color='red', linestyle='--', linewidth=2, label='JECFA THg Intake Threshold')

    ax.axhline(y=6424, color='#FF00FF', linestyle='--', linewidth=2, label='JECFA MeHg Intake Threshold')

    # Add labels, title, and legend
    ax.set_ylabel('Yearly THg Intake (µg)',fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xticks(x_pos + 0.175)
    ax.set_xticklabels(axis_labels, ha='center', fontsize=12)

    legend_labels = [
        'A-Non-Salmon Substitution',
        'B-Moose/Caribou Substitution',
        'C-Moderate Permafrost Thaw',
        'D-Severe Permafrost Thaw'
    ]

    handles, labels = ax.get_legend_handles_labels()
    extra_legend_handles = [mpatches.Patch(color='white', label=label) for label in legend_labels]


    all_handles = handles + extra_legend_handles
    all_labels = labels + legend_labels

    ax.legend(handles=all_handles, labels=all_labels, loc='upper left', title="Legend", title_fontsize=13, fontsize=12)


    plt.tight_layout()
    plt.show()

def absyearlydiff():
    ''' Creates absolute change bar plots for all the different simulations '''

    categories = ['A', 'B', 'C', 'D','A,C','B,C']
    coastal = [7250.41, -3161.81, 3037.58, 25097.26, 11383.13, -848.99]  # enter values from yearly_distribution function
    interior = [5865.73, -1963.76, 837.41, 7002.78, 7834.15, -1580.89]  # enter values from yearly_distribution function

    color1 = 'cornflowerblue'
    color2 = 'seagreen'

    # Define bar width and x locations
    x = np.arange(len(categories))
    width = 0.4

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars with custom colors
    ax.bar(x - width / 2, coastal, width, color=color1, edgecolor='black', linewidth=1, label='Coastal Yukon')
    ax.bar(x + width / 2, interior, width, color=color2, edgecolor='black', linewidth=1, label='Interior Yukon')

    # Add labels and legend
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Yearly THg Intake (µg)',fontsize=14)
    ax.set_title('Net Yearly THg Intake')
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=12)
    plt.show()

def main():
    # Functions to run for paper figures
    ''' 1. generate distribution data '''
    #### Coastal ####
    #proteinscenario(store_file_name='coastal protein 100 second draft.pkl',num_diets=100, region='Coastal')
    # permafrostthaw(store_file_name='coastal permafrost thaw 10k second draft.pkl',region='Coastal')
    # both(store_file_name='moderate coastal both 10k second draft.pkl',region='Coastal')

    #### Interior ####
    # proteinscenario(store_file_name='interior protein 10k second draft.pkl', region='Interior')
    # permafrostthaw(store_file_name='interior permafrost thaw 10k second draft.pkl',region='Interior')
    # both(store_file_name='moderate interior both 10k second draft.pkl',region='Interior')

    ''' 2. Use to extract peak data and export distribution plots'''
    #### protein substitution only ####
    # exportpeakdata(simulation='Protein',loadfile="coastal protein 10k second draft.pkl", region='Coastal',exportname="coastal diet survey first draft.csv", targets=[45,54])
    # exportpeakdata(simulation='Protein',loadfile="interior protein 10k first draft.pkl", region='Interior', exportname="interior diet survey first draft.csv", targets=[45,54])

    #### simulations with both moderate permafrost thaw and diet substitution ####
    # exportpeakdata(simulation='Protein', loadfile="moderate interior both 10k first draft.pkl", region='Moderate Interior',
                   #exportname="interior both diet survey.csv", targets=[45, 54])
    # exportpeakdata(simulation='Protein', loadfile="moderate coastal both 10k first draft.pkl",
                   #region='Moderate Coastal',exportname="interior both diet survey results.csv", targets=[45, 54])

    #### Permafrost thaw only ####
    #exportpeakdata(simulation='Permafrost Thaw',loadfile='interior permafrost thaw 10k second draft.pkl', region="Interior", exportname="interior permafrost thaw first draft.csv", targets=[43])
    # exportpeakdata(simulation='Permafrost Thaw',loadfile="coastal permafrost thaw 10k second draft.pkl", region='Coastal', exportname="coastal permafrost thaw first draft.csv", targets=[24,29,65.3])

    '''3. Yearly distribution monte carlo (skip these if going to be using yearly_distribution_data()'''
    # permafrost_thaw_yearly_distribution(file_name='interior with salmon 10k climate change.pkl',condition='Normal', num_samples=365, num_runs=10000)
    # proteinyearly_distribution(file_name='moderate interior both 10k first draft.pkl',proteins='Interior with Salmon', num_samples=365, num_runs=10000)

    ''' 4. Generates yearly bar graph and abs yearly differences '''
    #yearly_distributions_data()
    #absyearlydiff()



main()

