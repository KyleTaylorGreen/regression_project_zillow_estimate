README.md

## What:

> * Construct an ML Regression model that predict propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction in 2017.
> * Discover drivers or non-drivers of property tax value for single family properties
> * Deliver a report and recommendations based on the findings from above
> * Discover the location of the properties so Zach doesn't have to keep searching for that email.


## Why:

> * We want to make improvements on our current model for predicting property tax value
> * To help customers feel more confident in our product and helps us make better business decisions
How:

> * We're going to go through the data science pipeline by acquiring our data, preparing it, exploring it, and then modeling it.


## Hypotheses:

> * Square footage is positively related to taxdollarvalue count.
> * Bedroom and bathroom count are positively related to taxdollarvalue count.
> * Proximity to certain locations is related to taxdollarvalue count.


## Data Dictionary:

* parcelid: property identifier
* bedroomcnt: bedroom count of property
* bathroomcnt: bathroom count of property
* yearbuilt: property's year built
* fips: county of property
* calculatedfinishedsquarefeet: square footage of property
* latitude: latitude coordinate of property
* longitude: longitude coordinate of property


# Executive Summary - Findings & Next Steps

## Key Drivers - Square Feet, Bathroom Count, Proximity To Certain Locations

### Locations: Los Angeles County, Orange County, and Ventura County

## Model Performance:

>> Test RMSE of 278,000 dollars, improving on the last model by roughly 33%.

## Recommendations/Further Analysis:

> * Research other house properties to see what else could improve our model.
> * Try to research whether other distance-based features could be used, such as proximity to schools, to vacationing areas, etc.
> * Make linear regressions for each county, which will likely improve overall performance.