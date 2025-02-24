
(define (problem office-task)
    (:domain office-setup-domain)
    (:objects
        robot_1 - robot
        living_room_10 bathroom_1 bedroom_3 corridor_5 kitchen_9 - room
        book_into_the_shelf_38 folder_into_the_shelf_39 pen_into_the_shelf_40 notebook_into_the_shelf_41 - grabbable
        shelf_36 locker_37 desk_34 lamp_35 - movable
    )
    (:init
        (is-free robot_1)
        (at robot_1 living_room_10)
        (at shelf_36 bedroom_3)
        (at locker_37 kitchen_9)
        (at desk_34 kitchen_9)
        (at lamp_35 bathroom_1)
        (inside book_into_the_shelf_38 shelf_36)
        (inside folder_into_the_shelf_39 shelf_36)
        (inside pen_into_the_shelf_40 shelf_36)
        (inside notebook_into_the_shelf_41 shelf_36)
    )
    (:goal 
        (and 
            (at shelf_36 living_room_10)
            (at locker_37 living_room_10)
            (at desk_34 living_room_10)
            (at lamp_35 living_room_10)
            (forall (?item - grabbable)
                (not (inside ?item shelf_36))
            )
            (forall (?item - grabbable)
                (not (inside ?item locker_37))
            )
        )
    )
)
