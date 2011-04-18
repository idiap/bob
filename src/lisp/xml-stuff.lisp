;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(require :xmls)

#| TRASH
(defun collect-children(&rest childs)
  childs)
|#

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun create-attribute(name value)
  (list name value))

(defun collect-attributes(&rest attrs)
  attrs)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun create-array(string index)
  (xmls::make-node :name "array"
		   :attrs (collect-attributes 
			   (create-attribute "id" index))
		   :child string))

(defun create-external-array(string codec index)
  (xmls::make-node :name "external-array"
		   :attrs (collect-attributes 
			   (create-attribute "id" index)
			   (create-attribute "codec" codec)
			   (create-attribute "file" string))))

(defun make-array-set(contents &key (array-set-index 1) (role "Pattern") (elementtype "uint32") (shape ""))

  ;; create the arrayset
  (xmls::make-node :name  "arrayset"
		   :attrs (collect-attributes (create-attribute "id" array-set-index)
					      (create-attribute "role" role)
					      (create-attribute "elementtype" elementtype)
					      (create-attribute "shape" shape))
		   
		   :children (loop 
				for content being the elements of contents
				for index from 1
				collect (create-array content index))))

(defun create-dataset(lst-lst)
  (xmls::make-node :name "dataset"
		   :children (loop
				  for lst being the elements of lst-lst
				  for index from 1
				  collect (make-array-set lst :array-set-index index))))
  
  
(defun xml-to-file(tree filename)
  (with-open-file (stream filename
			  :direction :output
			  :if-exists :supersede)
    (xmls::write-xml tree stream :indent t)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defclass DatabaseSet()
  ()
  (:documentation "Abstract superclass to super all sets (array, external array and relations)"))

(defgeneric make-node(set set-index)
  (:documentation "create the xml node for this set (incl all its children)"))

(defclass ArraySet(DatabaseSet)
  ((role     :initarg :role    :initform "Pattern" :reader role)
   (etype    :initarg :etype   :initform "uint32"   :reader etype)
   (shape    :initarg :shape   :initform "1"       :reader shape)
   (contents :initarg :content :initform nil       :accessor contents)))

(defclass ExternalArraySet(ArraySet)
  ((codec :initarg :codec :initform "torch.image" :reader codec)))


(defmethod array-attrs(array-id (set ArraySet))
  (collect-attributes (create-attribute "id" array-id)
		      (create-attribute "role" (role set))
		      (create-attribute "elementtype" (etype set))
		      (create-attribute "shape" (shape set))))

(defmethod make-node((set ArraySet) array-set-index)
  (xmls::make-node :name  "arrayset"
		   :attrs (array-attrs array-set-index set) 
		   :children (loop 
				for content being the elements of (contents set)
				for index from 1
				collect (create-array content index))))

(defmethod make-node((set ExternalArraySet) array-set-index)
  (xmls::make-node :name  "arrayset"
		   :attrs (array-attrs array-set-index set) 
		   :children (loop 
				for content being the elements of (contents set)
				for index from 1
				collect (create-external-array content (codec set) index))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun make-dataset(&rest arrays)
  (xmls::make-node :name "dataset"
		   :children (loop
				for array being the elements of arrays
				for index from 0
				collect (make-node array index))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun make-member(array-id arrayset-id)
  (xmls::make-node :name "member" 
		   :attrs (collect-attributes (create-attribute "array-id" array-id)
					      (create-attribute "arrayset-id" arrayset-id))))

(defun column-index->table-index(column)
  (1+ column))

(defun make-self(id column)
  (make-member id (column-index->table-index column)))

(defun id->relations(this-id id-data column)
  (let ((children (loop for array-id in id-data
		     collect (make-member array-id column))))

  (xmls::make-node :name "relation"
		   :attrs (collect-attributes (create-attribute "id" this-id))
		   :children (push (make-self this-id column) children))))

(defun create-role(role &key (min 1) (max 0))
  (xmls::make-node :name "role"
		   :attrs (collect-attributes (create-attribute "arrayset-role" role)
					      (create-attribute "min" min)
					      (create-attribute "max" max))))
				    
(defun create-relationset(column column-ids column-data &key (set-name "pattern_target"))
  (let ((children (loop
		     for index from 0
		     for id from 1 to (length column-ids)
		     collect (id->relations id (nth index column-data) (column-index->table-index column)))))

    (push (create-role "TargetXYZ") children)
    (push (create-role "PatternXYZ") children)

    (xmls::make-node :name "relationset"
		     :attrs (collect-attributes (create-attribute "name" set-name))
		     :children children)))

(defun create-relationset-compact(column chunk &key (set-name "pattern_target"))
  "Makes it easier to call if we can lump stuff together"
  (create-relationset column (first chunk) (second chunk) :set-name set-name))


(defclass Relationset(DatabaseSet)
  ((column :initarg :column :initform 0 :reader column)
   (chunk  :initarg :chunk  :initform nil :reader chunk)
   (name   :initarg :name   :initform "pattern_target" :reader name)
   (role-1 :initarg :role-1 :initform "pattern" :reader role-1)
   (role-2 :initarg :role-2 :initform "target"  :reader role-2)))

(defmethod make-node((set Relationset) index)
  (declare (ignore index))
  (create-relationset-compact (column set) (chunk set) :set-name (name set)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defclass Karlek(DatabaseSet)
  ((name    :initarg :name    :initform "my-karlek" :reader name)
   (content :initarg :content :initform nil         :reader content)))

(defmethod make-node((Set Karlek) index)
  (declare (ignore index))
  (xmls::make-node :name "karlek"
		   :attrs (collect-attributes (create-attribute "name" (name set)))
		   :children (loop for relation in (content set)
				collect (xmls::make-node :name "member"
							 :child relation))))

(defclass Location(DatabaseSet)
  ((pathnames :initarg :pathnames :initform "" :reader pathnames)))

(defmethod make-node((set Location) index)
  (declare (ignore index))
  (xmls::make-node :name "pathlist"
		   :children (loop for pathname in (pathnames set)
				  collect (xmls::make-node :name "entry"
							   :attrs (collect-attributes (create-attribute "path" pathname))))))
